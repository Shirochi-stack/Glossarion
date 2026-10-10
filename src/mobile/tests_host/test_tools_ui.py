"""Host tests for the U6 Tools: job kinds (QA scan, validate / rename, headers, metadata), the
screen models and the Tools screens (hub, QA Scanner, QA report viewer, Converter, Headers &
metadata) in the fake Flet session.

* Job adapters run against fake ``ctx`` objects and fake shared modules (``qa_scan_runtime``,
  ``translate_headers_standalone``, ``TransateKRtoEN``, ``output_naming``); one QA test runs the
  real shared scanner on a fixture workspace (thread executor, isolated CONFIG_FILE / HOME /
  OUTPUT_DIRECTORY), and the registered kinds run through the real ``JobService`` with the U3
  ``FakeBackend``.
* Models: the QA mode cards and the metadata field tables are compared with the desktop
  literals (AST of ``QA_Scanner_GUI.py`` / ``metadata_batch_translator.py``); Custom-mode
  conversions, report summaries and the WebView report HTML; header / TOC cache deletion with
  the RECYCLED link on a fixture workspace; font / CSS imports; job specs.
* Screens: built in the in-memory Flet session from ``test_bootstrap`` (``page.update``
  serialises every control); dialogs are scripted through ``ctx.extras["answers"]``. A reopened
  QA / Converter / Headers screen follows the queued or running job an earlier visit started.

Real data is never touched: every Library / output / config path is a pytest tmp dir.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_tools_ui.py
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import os
import sys
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
from glossarion_mobile.job_kinds import compile as compile_kind  # noqa: E402
from glossarion_mobile.job_kinds import headers as headers_kind  # noqa: E402
from glossarion_mobile.job_kinds import metadata as metadata_kind  # noqa: E402
from glossarion_mobile.job_kinds import qa as qa_kind  # noqa: E402
from glossarion_mobile.services.jobs import JobKind, JobState  # noqa: E402
from glossarion_mobile.ui.tools import compile_model as cm  # noqa: E402
from glossarion_mobile.ui.tools import headers_model as hm  # noqa: E402
from glossarion_mobile.ui.tools import qa_model as qm  # noqa: E402
from glossarion_mobile.ui.tools import targets as tg  # noqa: E402


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==========================================================================
# Fakes
# ==========================================================================


class FakeCtx:
    """What a job adapter gets (JobContext surface)."""

    def __init__(self, owner, *, params=None, inputs=(), config=None) -> None:
        self.owner = owner
        self.params = dict(params or {})
        self.inputs = tuple(inputs)
        self.config = dict(config or {})
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


def fake_qa_runtime(*, defaults=None, match=None):
    """A stand-in ``qa_scan_runtime`` that records every ``run_qa_scan_path`` call."""
    module = types.ModuleType("qa_scan_runtime")
    base = {"check_word_count_ratio": True, "check_ai_truncation_detection": False, "check_silent_truncation": False,
            "warn_name_mismatch": True, "cache_show_stats": False}
    base.update(defaults or {})
    module.calls = []
    module.normalize_qa_scan_settings = lambda settings=None, target_language=None: {**base, **dict(settings or {})}
    module.is_direct_text_qa_path = lambda path: bool(path) and "Direct Text" in str(path)
    module.check_epub_folder_match = match or (lambda epub, folder, suffixes="": epub == folder)
    module.DEFAULT_CUSTOM_MODE_SETTINGS = {"similarity": 85, "semantic": 80, "structural": 90, "word_overlap": 75,
                                           "minhash_threshold": 80, "consecutive_chapters": 2,
                                           "check_all_pairs": False, "sample_size": 3000, "min_text_length": 500,
                                           "min_duplicate_word_count": 500}
    module.automatic_qa_output_candidates = lambda source, **kw: [
        os.path.join(kw.get("output_root") or "", os.path.splitext(os.path.basename(source))[0])]

    def run_qa_scan_path(folder, log=print, stop_flag=None, mode="quick-scan", qa_settings=None, epub_path=None,
                         selected_files=None, text_file_mode=None, progress_path=None, owner=None, config=None,
                         allow_direct_text=False, **kw):
        # devfix 2026-10-08: the shared loop hands on the chat-QA opt-in (allow_direct_text)
        module.calls.append({"folder": folder, "mode": mode, "settings": dict(qa_settings or {}),
                             "epub": epub_path, "owner": owner, "stopped": bool(stop_flag and stop_flag()),
                             "allow_direct_text": allow_direct_text})
        report = qa_kind.report_path_for(folder)
        os.makedirs(os.path.dirname(report), exist_ok=True)
        Path(report).write_text("<html><body>report</body></html>", encoding="utf-8")
        return []

    module.run_qa_scan_path = run_qa_scan_path
    # U7: the adapter runs the real shared loop / settings loader / stop escalation over these
    # recording primitives (functions rebound to this module's globals); the flag setters record.
    import qa_scan_runtime as real

    module.os = os
    for name in ("run_bulk_qa_scan", "load_current_qa_settings", "next_qa_stop_phase"):
        fn = getattr(real, name)
        bound = types.FunctionType(fn.__code__, module.__dict__, fn.__name__, fn.__defaults__, fn.__closure__)
        bound.__kwdefaults__ = fn.__kwdefaults__
        setattr(module, name, bound)
    module.flags = []

    def _stop_scan():
        module.flags.append("stop_scan")
        stop = getattr(sys.modules.get("scan_html_folder"), "stop_scan", None)
        if callable(stop):
            stop()

    module.reset_qa_cancel_flags = lambda: module.flags.append("reset")
    module.apply_qa_graceful_stop_flags = lambda: (module.flags.append("graceful"), _stop_scan())
    module.apply_qa_force_stop_flags = lambda: (module.flags.append("force"), _stop_scan())
    module.clear_qa_stop_flags = lambda: module.flags.append("clear")
    return module


def make_epub(path: Path, chapters=("ch001.xhtml", "ch002.xhtml"), title="Raw Book", extra_meta="") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml",
                    '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:'
                    'container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-'
                    'package+xml"/></rootfiles></container>')
        manifest = "".join(f'<item id="c{i}" href="{n}" media-type="application/xhtml+xml"/>'
                           for i, n in enumerate(chapters))
        spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
        zf.writestr("OEBPS/content.opf",
                    '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata '
                    f'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title><dc:creator>Author'
                    '</dc:creator><dc:language>ko</dc:language><dc:subject>Fantasy</dc:subject>'
                    f'{extra_meta}</metadata><manifest>{manifest}</manifest><spine>{spine}</spine></package>')
        for n in chapters:
            zf.writestr(f"OEBPS/{n}", f"<html><head><title>{n}</title></head><body><h1>{n}</h1><p>text</p>"
                        "</body></html>")
    return str(path)


def tool_target(tmp_path: Path, name="Book", *, source=True, folder=True, **kwargs) -> tg.ToolTarget:
    out = tmp_path / "Output" / name
    if folder:
        out.mkdir(parents=True, exist_ok=True)
        (out / "response_ch001.html").write_text("<html><body><p>one</p></body></html>", encoding="utf-8")
    raw = make_epub(tmp_path / "Library" / "Raw" / f"{name}.epub") if source else ""
    return tg.ToolTarget(title=name, folder=str(out) if folder else "", source=raw, kind="epub", **kwargs)


# ==========================================================================
# Registration
# ==========================================================================


def test_u6_kinds_are_registered_and_enable_the_library_metadata_actions():
    for kind, module in (("qa_scan", "qa"), ("validate_epub", "compile"), ("rename_outputs", "compile"),
                         ("translate_headers", "headers"), ("metadata", "metadata")):
        assert job_kinds.KIND_MODULES[kind] == module
        assert JobKind(kind).value == kind
        info = job_kinds.get_kind(kind)
        assert info.stop_kind == "translation" and callable(info.run)
    assert job_kinds.get_kind("validate_epub").resumable is False
    from glossarion_mobile.services.library import LibraryService

    class Jobs:
        def has_kind(self, kind):
            return job_kinds.get_kind(kind) is not None

    service = LibraryService(jobs=Jobs(), config={})
    # U5 hand-off: the selection bar / Book page "Translate Metadata" check exactly this
    assert service.has_job_kind("metadata") is True


# ==========================================================================
# QA scan adapter
# ==========================================================================


def test_qa_adapter_bulk_scan_rules_and_reports(tmp_path, monkeypatch):
    fake = fake_qa_runtime()
    monkeypatch.setitem(sys.modules, "qa_scan_runtime", fake)
    with_source = tool_target(tmp_path, "Alpha")
    no_source = tool_target(tmp_path, "Beta", source=False)
    chat = tmp_path / "Output" / "Direct Text" / "chat1"
    chat.mkdir(parents=True)
    owner = types.SimpleNamespace(config={})
    ctx = FakeCtx(owner, params={"mode": "aggressive", "targets": [with_source.to_param(), no_source.to_param(),
                                                                    {"folder": str(chat), "source": None},
                                                                    {"folder": str(tmp_path / "missing")}]},
                  config={"qa_scanner_settings": {"check_ai_truncation_detection": True}, "output_language": "English"})
    result = qa_kind.run(ctx)
    assert result["ok"] is True
    assert [os.path.basename(c["folder"]) for c in fake.calls] == ["Alpha", "Beta"]
    alpha, beta = fake.calls
    assert alpha["mode"] == "aggressive" and alpha["epub"] == with_source.source and alpha["owner"] is owner
    assert alpha["settings"]["check_word_count_ratio"] and alpha["settings"]["check_ai_truncation_detection"]
    # bulk + no source: every source-dependent check is off for that folder (desktop rule)
    assert not any(beta["settings"][k] for k in qa_kind.SOURCE_DEPENDENT_CHECKS) and beta["epub"] is None
    assert any("Skipping Direct Text folder" in line for line in ctx.logs)
    assert any("Ignoring missing output folder" in line for line in ctx.logs)
    assert any("Starting bulk QA scan in AGGRESSIVE mode for 2 folders" in line for line in ctx.logs)
    assert any("No matching EPUB found for folder 'Beta'" in line for line in ctx.logs)
    assert any("Matched from selected files: Alpha.epub" in line for line in ctx.logs)  # Library source by folder
    assert fake.flags == ["reset"] and "Scanning 2/2" in ctx.phases
    assert any("Bulk scan summary: 2 successful, 0 failed" in line for line in ctx.logs)
    assert ctx.logs[-1] == "✅ Bulk QA scan completed."
    assert result["outputs"] == ctx.outputs == [qa_kind.report_path_for(with_source.folder),
                                                qa_kind.report_path_for(no_source.folder)]
    assert ctx.results["qa_reports"] == result["outputs"] and ctx.results["qa_mode"] == "aggressive"


def test_qa_adapter_single_folder_word_count_off_mismatch_and_stop(tmp_path, monkeypatch):
    fake = fake_qa_runtime()
    monkeypatch.setitem(sys.modules, "qa_scan_runtime", fake)
    target = tool_target(tmp_path, "Gamma")
    other_source = make_epub(tmp_path / "Library" / "Raw" / "Different.epub")
    ctx = FakeCtx(object(), params={"mode": "quick-scan", "disable_word_count": True,
                                    "targets": [{"folder": target.folder, "source": other_source}]})
    assert qa_kind.run(ctx)["ok"] is True
    assert fake.calls[-1]["settings"]["check_word_count_ratio"] is False
    assert not any("name mismatch" in line for line in ctx.logs)  # word count off: no mismatch warning
    ctx2 = FakeCtx(object(), params={"targets": [{"folder": target.folder, "source": other_source}]})
    qa_kind.run(ctx2)
    assert any("Warning: source/folder name mismatch - Different vs Gamma" in line for line in ctx2.logs)
    # a stop before the first folder: nothing scanned, scan_html_folder.stop_scan raised once
    stopped = []
    monkeypatch.setitem(sys.modules, "scan_html_folder", types.SimpleNamespace(stop_scan=lambda: stopped.append(1)))
    ctx3 = FakeCtx(object(), params={"targets": [{"folder": target.folder}]})
    ctx3.stop = True
    calls_before = len(fake.calls)
    assert qa_kind.run(ctx3)["ok"] is None
    assert len(fake.calls) == calls_before and stopped == [1]
    assert fake.flags[-3:] == ["force", "stop_scan", "clear"]
    assert any("Bulk scan stopped by user at folder 1/1" in line for line in ctx3.logs)
    with pytest.raises(Exception, match="Unknown QA scan mode"):
        qa_kind.run(FakeCtx(object(), params={"mode": "nope", "targets": [{"folder": target.folder}]}))


def test_qa_adapter_runs_the_real_shared_scanner(tmp_path, monkeypatch):
    try:
        import qa_scan_runtime  # noqa: F401
        import scan_html_folder  # noqa: F401
    except Exception as exc:  # pragma: no cover - a bundle without the scanner's dependencies
        pytest.skip(f"the shared scanner is not importable ({exc})")
    for name, value in (("GLOSSARION_NO_PROCESSES", "1"), ("CONFIG_FILE", str(tmp_path / "config.json")),
                        ("HOME", str(tmp_path / "home")), ("USERPROFILE", str(tmp_path / "home")),
                        ("OUTPUT_DIRECTORY", str(tmp_path / "Output")),
                        ("GLOSSARION_LIBRARY_DIR", str(tmp_path / "Library"))):
        monkeypatch.setenv(name, value)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    folder = tmp_path / "Output" / "Real"
    folder.mkdir(parents=True)
    text = "The knight walked into the hall and greeted everyone warmly. " * 30
    for index in range(1, 4):
        body = text if index != 2 else text + " 이것은 번역되지 않은 한국어 문장입니다 그리고 더 많은 한국어 "
        (folder / f"response_{index:04d}_ch{index}.html").write_text(
            f"<html><head><title>Chapter {index}</title></head><body><h1>Chapter {index}</h1><p>{body}</p>"
            f"</body></html>", encoding="utf-8")
    owner = types.SimpleNamespace(config={"qa_scanner_settings": {"check_word_count_ratio": False}})
    ctx = FakeCtx(owner, params={"mode": "quick-scan", "targets": [{"folder": str(folder)}]},
                  config={"qa_scanner_settings": {"check_word_count_ratio": False, "check_missing_header_tags": False},
                          "output_language": "English"})
    result = qa_kind.run(ctx)
    report = qa_kind.report_path_for(str(folder))
    assert result["ok"] is True and result["outputs"] == [report] and os.path.isfile(report)
    summary = qm.load_report_summary(report)
    assert summary is not None and summary.total == 3
    assert ctx.logs[-1] == "✅ QA scan completed successfully."


def test_chat_qa_scans_a_direct_text_workspace_with_the_real_scanner(tmp_path, monkeypatch):
    """Owner device report #7: a QA scan from the chat runs Quick Scan with the duplicate check off (sample
    size 0) on the chat's own Direct Text workspace, through the shared scanner (threads, isolated env)."""
    try:
        import qa_scan_runtime  # noqa: F401
        import scan_html_folder  # noqa: F401
    except Exception as exc:  # pragma: no cover - a bundle without the scanner's dependencies
        pytest.skip(f"the shared scanner is not importable ({exc})")
    for name, value in (("GLOSSARION_NO_PROCESSES", "1"), ("CONFIG_FILE", str(tmp_path / "config.json")),
                        ("HOME", str(tmp_path / "home")), ("USERPROFILE", str(tmp_path / "home")),
                        ("APPDATA", str(tmp_path / "appdata")), ("GLOSSARION_HTTP_LOG", "0"),
                        ("OUTPUT_DIRECTORY", str(tmp_path / "Output")),
                        ("GLOSSARION_LIBRARY_DIR", str(tmp_path / "Library"))):
        monkeypatch.setenv(name, value)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    text = "The knight walked into the hall and greeted everyone warmly. " * 30

    def workspace(folder: Path) -> Path:
        folder.mkdir(parents=True)
        for index in range(1, 4):
            (folder / f"response_{index:04d}_ch{index}.html").write_text(
                f"<html><head><title>Chapter {index}</title></head><body><h1>Chapter {index}</h1><p>{text}</p>"
                f"</body></html>", encoding="utf-8")
        return folder

    chat = workspace(tmp_path / "Output" / "Direct Text" / "Novel - c1" / "Attachments" / "Book")
    saved = {"qa_scanner_settings": {"check_missing_header_tags": False}, "output_language": "English"}
    kind, _title, inputs, params, _origin = qm.chat_qa_job(str(chat), None, cid="c1", chat_title="Novel")
    owner = types.SimpleNamespace(config=dict(saved))
    ctx = FakeCtx(owner, params=params, inputs=inputs, config=saved)
    result = qa_kind.run(ctx)
    report = qa_kind.report_path_for(str(chat))
    assert kind == "qa_scan" and result["ok"] is True and result["outputs"] == [report] and os.path.isfile(report)
    assert report == str(chat / "Book_Scan Report" / "validation_results.html")
    assert any("duplicate detection disabled (sample size set to 0)" in line for line in ctx.logs)
    assert not any("QA scan skipped" in line or "Skipping Direct Text" in line for line in ctx.logs)
    assert qm.load_report_summary(report).total == 3
    # the same folder through the inputs fallback (never flagged): the desktop Direct Text rule
    ctx2 = FakeCtx(owner, params={"mode": "quick-scan"}, inputs=(str(chat),), config=saved)
    assert qa_kind.run(ctx2)["ok"] is False
    # a chat book the Library already holds (auto-migrated): no flag, and a saved 1000 runs the duplicate pass
    library = workspace(tmp_path / "Output" / "Book2")
    _k, _t, lib_inputs, lib_params, _o = qm.chat_qa_job(str(library), None, cid="c1", chat_title="Novel")
    assert "direct_text" not in lib_params["targets"][0]
    with_1000 = {**saved, "qa_scanner_settings": {**saved["qa_scanner_settings"], "quick_scan_sample_size": 1000}}
    ctx3 = FakeCtx(owner, params=lib_params, inputs=lib_inputs, config=with_1000)
    assert qa_kind.run(ctx3)["ok"] is True and os.path.isfile(qa_kind.report_path_for(str(library)))
    assert not any("duplicate detection disabled" in line for line in ctx3.logs)


def test_qa_scan_through_the_job_service(tmp_path, monkeypatch):
    tj = _load("_glossarion_tools_jobs_helpers", "test_jobs.py")
    fake = fake_qa_runtime()
    monkeypatch.setitem(sys.modules, "qa_scan_runtime", fake)
    target = tool_target(tmp_path, "Delta")
    service, backend = tj.make_service(tmp_path)
    job_id = service.submit(qm.qa_spec([target], "ai-hunter"))
    assert service.wait_idle(tj.TIMEOUT)
    snap = service.snapshot(job_id)
    assert snap.state is JobState.DONE and snap.kind == "qa_scan"
    assert snap.outputs == (qa_kind.report_path_for(target.folder),)
    assert snap.result["qa_mode"] == "ai-hunter" and ("reset", "translation") in backend.events
    assert fake.calls[-1]["owner"] is backend.owners[-1]
    service.close()


# ---- devfix 2026-10-08 (owner device report #7): QA scan from the chat, Quick Scan, sample size 0 ----


def _chat_workspace(tmp_path: Path, chat="Novel - c1", name="Book") -> Path:
    """``<output root>/Direct Text/<chat>/Attachments/<book>`` (direct_text_store's attachment workspace)."""
    folder = tmp_path / "Output" / "Direct Text" / chat / "Attachments" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "response_ch001.html").write_text("<html><body><p>one</p></body></html>", encoding="utf-8")
    return folder


def test_qa_adapter_scans_flagged_chat_workspaces_only(tmp_path, monkeypatch):
    fake = fake_qa_runtime()
    monkeypatch.setitem(sys.modules, "qa_scan_runtime", fake)
    chat = _chat_workspace(tmp_path)
    source = make_epub(chat.parent / "Book.epub")  # the chat's attachment copy (a Direct Text path too)
    other = _chat_workspace(tmp_path, "Novel - c2", "Other")
    kind, _title, inputs, params, _origin = qm.chat_qa_job(str(chat), source, cid="c1", chat_title="Novel")
    params["targets"].append({"folder": str(other), "source": None})  # an unflagged Direct Text folder
    ctx = FakeCtx(object(), params=params, inputs=inputs + (str(other),), config={"output_language": "English"})
    result = qa_kind.run(ctx)
    assert kind == "qa_scan" and result["ok"] is True
    # only the flagged chat workspace is scanned, with its source and the shared loop's opt-in
    assert [os.path.basename(c["folder"]) for c in fake.calls] == ["Book"]
    assert fake.calls[0]["allow_direct_text"] is True and fake.calls[0]["epub"] == os.path.abspath(source)
    assert fake.calls[0]["mode"] == "quick-scan"
    assert any(line == f"⏭️ Skipping Direct Text folder during QA scan: {other}" for line in ctx.logs)
    assert result["outputs"] == [qa_kind.report_path_for(str(chat))]
    # the inputs fallback is never flagged: the desktop skip and its log line
    ctx2 = FakeCtx(object(), params={"mode": "quick-scan"}, inputs=(str(chat),))
    assert qa_kind.run(ctx2)["ok"] is False and len(fake.calls) == 1
    assert any("Skipping Direct Text folder during QA scan" in line for line in ctx2.logs)
    assert ctx2.logs[-1] == "⏭️ QA scan skipped: no non-Direct-Text output folders were selected."
    # a flag on a folder the Library holds (an auto-migrated chat book) needs no opt-in
    library = tool_target(tmp_path, "Lib")
    ctx3 = FakeCtx(object(), params={"targets": [{"folder": library.folder, "source": library.source,
                                                  "direct_text": True}]})
    assert qa_kind.run(ctx3)["ok"] is True and fake.calls[-1]["allow_direct_text"] is False
    # Tools › QA Scanner specs are unchanged: no flag, no opt-in
    assert qm.qa_spec([library], "quick-scan").params["targets"] == [library.to_param()]
    assert qa_kind.normalize_targets(params, ()) == [(str(chat), os.path.abspath(source)), (str(other), None)]
    assert qa_kind.normalize_targets(params, (), with_flags=True)[0][2] is True


def test_qa_adapter_mobile_sample_size_default_survives_the_per_folder_reload(tmp_path, monkeypatch):
    """Owner: Quick Scan sample size 0 (duplicate check off) on mobile, not the desktop 1000; a saved value
    (Tools › QA Scanner / Settings) still wins. The per-folder settings reload keeps it; config is never written."""
    fake = fake_qa_runtime(defaults={"quick_scan_sample_size": 1000, "check_word_count_ratio": False})
    monkeypatch.setitem(sys.modules, "qa_scan_runtime", fake)
    a, b = tool_target(tmp_path, "A"), tool_target(tmp_path, "B")
    config = {"qa_scanner_settings": {"check_ai_truncation_detection": False}, "output_language": "English"}
    before = json.loads(json.dumps(config))
    ctx = FakeCtx(object(), params={"mode": "quick-scan", "targets": [a.to_param(), b.to_param()]}, config=config)
    assert qa_kind.run(ctx)["ok"] is True
    assert [c["settings"]["quick_scan_sample_size"] for c in fake.calls] == [0, 0]
    assert ctx.config == before and "quick_scan_sample_size" not in ctx.config["qa_scanner_settings"]
    assert ("⚡ Quick Scan duplicate check sample size: 0 (duplicate check off) · Glossarion Mobile default"
            in ctx.logs)
    for saved in (1000, 250, -1):
        fake.calls.clear()
        ctx = FakeCtx(object(), params={"mode": "quick-scan", "targets": [a.to_param(), b.to_param()]},
                      config={"qa_scanner_settings": {"quick_scan_sample_size": saved}})
        qa_kind.run(ctx)
        assert [c["settings"]["quick_scan_sample_size"] for c in fake.calls] == [saved, saved]
        assert f"⚡ Quick Scan duplicate check sample size: {saved}" in ctx.logs
    # other modes scan with the same settings and no Quick Scan line
    fake.calls.clear()
    ctx = FakeCtx(object(), params={"mode": "aggressive", "targets": [a.to_param()]}, config={})
    qa_kind.run(ctx)
    assert fake.calls[0]["settings"]["quick_scan_sample_size"] == 0
    assert not any("Quick Scan duplicate check" in line for line in ctx.logs)
    assert qa_kind.with_mobile_qa_defaults({"quick_scan_sample_size": 1000}, {}) == {"quick_scan_sample_size": 0}
    assert qa_kind.with_mobile_qa_defaults({"quick_scan_sample_size": 1000},
                                           {"qa_scanner_settings": {"quick_scan_sample_size": 1000}}) == {
        "quick_scan_sample_size": 1000}


def test_chat_qa_job_parts_summary_and_the_job_service(tmp_path, monkeypatch):
    tj = _load("_glossarion_tools_jobs_helpers", "test_jobs.py")
    fake = fake_qa_runtime()
    monkeypatch.setitem(sys.modules, "qa_scan_runtime", fake)
    chat = _chat_workspace(tmp_path)
    parts = qm.chat_qa_job(str(chat), None, cid="c1", chat_title="Novel")
    kind, title, inputs, params, origin = parts
    assert (kind, title, inputs) == ("qa_scan", "Book", (str(chat),))
    # Quick Scan; the sample size is the saved / mobile one the job reads (no per-run override)
    assert params == {"mode": qm.CHAT_QA_MODE, "disable_word_count": True,
                      "targets": [{"folder": str(chat), "source": None, "direct_text": True}]}
    assert qm.CHAT_QA_MODE == "quick-scan" and qm.CHAT_QA_SAMPLE_SIZE == 0 == qa_kind.MOBILE_QUICK_SAMPLE_SIZE
    # a chat origin without params["chat_id"]: the chat's JobStrip shows the job (no chat card)
    assert origin == {"type": "chat", "cid": "c1", "label": "Chat · Novel"} and "chat_id" not in params
    library = tool_target(tmp_path, "Lib")
    _k, _t, _i, lib_params, lib_origin = qm.chat_qa_job(library.folder, library.source, cid="c1", chat_title="")
    assert lib_params == {"mode": "quick-scan", "targets": [library.to_param()]} and lib_origin["label"] == "Chat"
    assert qm.chat_qa_job("", None, cid="c1", chat_title="Novel") == "No output folder to scan"
    assert qm.chat_qa_summary_line() == "Quick Scan · duplicate check off (sample size 0)"
    assert qm.chat_qa_summary_line({}) == "Quick Scan · duplicate check off (sample size 0)"
    assert (qm.chat_qa_summary_line({"qa_scanner_settings": {"quick_scan_sample_size": 1000}})
            == "Quick Scan · duplicate check sample size 1000")
    assert qm.sample_size_text(-1) == "duplicate check on the full text (sample size -1)"
    # the parts run through the real JobService (the chat's env.jobs.submit builds this JobSpec)
    from glossarion_mobile.services.jobs import JobSpec

    service, _backend = tj.make_service(tmp_path)
    job_id = service.submit(JobSpec(kind=kind, title=title, inputs=inputs, params=params, origin=origin))
    assert service.wait_idle(tj.TIMEOUT)
    snap = service.snapshot(job_id)
    assert snap.state is JobState.DONE and snap.outputs == (qa_kind.report_path_for(str(chat)),)
    assert fake.calls[-1]["allow_direct_text"] is True and fake.calls[-1]["settings"]["check_word_count_ratio"] is False
    service.close()
    # a build whose shared scanner lacks the opt-in: a chat workspace waits for the Library
    old = fake_qa_runtime()

    def old_loop(folders_to_scan, *, mode, epub_path, qa_settings, load_settings, selected_mode_value,
                 disable_word_count_for_run, epub_basename_map, global_selected_files, log, stop_flag,
                 owner=None, on_report=None):  # the U9 signature: no opt-in
        return 0, 0

    old.run_bulk_qa_scan = old_loop
    monkeypatch.setitem(sys.modules, "qa_scan_runtime", old)
    assert qm.chat_qa_job(str(chat), None, cid="c1", chat_title="Novel") == qm.CHAT_QA_UNAVAILABLE
    assert isinstance(qm.chat_qa_job(library.folder, None, cid="c1", chat_title="Novel"), tuple)
    # a wrapper that forwards **kwargs (instrumentation, a decorator without functools.wraps) passes it on
    wrapped = fake_qa_runtime()
    inner = wrapped.run_bulk_qa_scan
    wrapped.run_bulk_qa_scan = lambda *args, **kwargs: inner(*args, **kwargs)
    monkeypatch.setitem(sys.modules, "qa_scan_runtime", wrapped)
    assert isinstance(qm.chat_qa_job(str(chat), None, cid="c1", chat_title="Novel"), tuple)


def test_quick_sample_size_migration_runs_once_and_only_on_the_desktop_1000(tmp_path):
    from glossarion_mobile.state.config_store import MobileConfigStore
    from glossarion_mobile.state.prefs import Prefs

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": "m", "qa_scanner_settings": {"quick_scan_sample_size": 1000,
                                                                           "min_file_length": 5}}),
                           encoding="utf-8")
    store = MobileConfigStore(config_path, debounce=0.01)
    prefs = Prefs(tmp_path / "mobile_state.json")
    store.load()
    prefs.load()
    try:
        assert qm.migrate_quick_sample_size(store.get, store.set, prefs) is True
        assert store.get(qm.QUICK_SAMPLE_KEY) == 0 and qm.quick_sample_size(store.snapshot()) == 0
        assert prefs.get(qm.QUICK_SAMPLE_MIGRATION_PREF) is True
        store.set(qm.QUICK_SAMPLE_KEY, 1000)  # the owner types 1000 again later: it stays
        assert qm.migrate_quick_sample_size(store.get, store.set, prefs) is False
        assert store.get(qm.QUICK_SAMPLE_KEY) == 1000
        store.flush()
        prefs.flush()
    finally:
        store.close()
        prefs.close()
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved == {"model": "m", "qa_scanner_settings": {"quick_scan_sample_size": 1000, "min_file_length": 5}}
    assert json.loads((tmp_path / "mobile_state.json").read_text(encoding="utf-8"))[qm.QUICK_SAMPLE_MIGRATION_PREF]
    # only exactly 1000 is migrated; nothing saved stays unsaved (the job uses the mobile default)
    for value in (500, -1, 0, "1000", 1000.5, True, None):
        data = {} if value is None else {"qa_scanner_settings": {"quick_scan_sample_size": value}}
        flags = FakePrefs()

        def get(key, default=None, data=data):
            return (data.get(key[0]) or {}).get(key[1], default)

        def put(key, value, data=data):
            data.setdefault(key[0], {})[key[1]] = value

        assert qm.migrate_quick_sample_size(get, put, flags) is False, value
        assert data == ({} if value is None else {"qa_scanner_settings": {"quick_scan_sample_size": value}})
        assert flags.get(qm.QUICK_SAMPLE_MIGRATION_PREF) is True
    assert qm.migrate_quick_sample_size(lambda k, d=None: 1000, lambda k, v: None, None) is False  # no Prefs
    assert qm.quick_sample_size({}) == 0 and qm.quick_sample_size(None) == 0
    assert qm.quick_sample_size({"qa_scanner_settings": {"quick_scan_sample_size": 300}}) == 300


# ==========================================================================
# Metadata adapter
# ==========================================================================


class MetadataOwner:
    def __init__(self, out_root: str) -> None:
        self.out_root = out_root
        self.config = {"batch_translation": True}
        self.batch_translation_var = True
        self.seen = {}

    def _resolve_translation_output_dir(self, path):
        return os.path.join(self.out_root, os.path.splitext(os.path.basename(path))[0])

    def _prepare_translation_run(self, files):
        self.seen = {"files": list(files), "metadata_only": self._metadata_only_run,
                     "roots": dict(self._metadata_output_roots), "selected": list(self.selected_files),
                     "single": self._single_chapter_filter, "stream": self._force_stream_all}
        return {"files": list(files)}

    def _translation_worker(self, request):
        return None


def test_metadata_adapter_sets_the_desktop_metadata_run(tmp_path):
    a = make_epub(tmp_path / "raw" / "A.epub")
    b = make_epub(tmp_path / "raw" / "B.epub")
    owner = MetadataOwner(str(tmp_path / "Output"))
    folders = [str(tmp_path / "Output" / "A"), str(tmp_path / "Elsewhere" / "B")]
    ctx = FakeCtx(owner, inputs=(a, b, a, str(tmp_path / "raw" / "missing.epub")),
                  params={"output_roots": {a: str(tmp_path / "Output"), b: str(tmp_path / "Elsewhere")}})
    result = metadata_kind.run(ctx)
    assert result["ok"] is None
    assert owner.seen["metadata_only"] is True and owner.seen["selected"] == [os.path.abspath(a), os.path.abspath(b)]
    assert owner.seen["roots"] == {os.path.normcase(os.path.abspath(a)): os.path.abspath(str(tmp_path / "Output")),
                                   os.path.normcase(os.path.abspath(b)): os.path.abspath(str(tmp_path / "Elsewhere"))}
    assert owner.seen["single"] is None and owner.seen["stream"] is False
    assert ctx.logs[0] == "🌐 Queued metadata translation for 2 EPUBs (thread-pool batch)"
    assert owner._metadata_only_run is False and owner._metadata_output_roots == {}
    # LibraryService.metadata_spec shape: a list of book output folders aligned with the inputs
    assert metadata_kind.output_roots_for(folders, (a, b)) == {
        os.path.normcase(os.path.abspath(a)): os.path.abspath(str(tmp_path / "Output")),
        os.path.normcase(os.path.abspath(b)): os.path.abspath(str(tmp_path / "Elsewhere"))}
    assert metadata_kind.output_roots_for(folders[:1], (a, b)) == {}  # misaligned: no override
    with pytest.raises(Exception, match="no valid EPUB"):
        metadata_kind.run(FakeCtx(owner, inputs=(str(tmp_path / "x.txt"),)))


# ==========================================================================
# Headers adapter
# ==========================================================================


class HeadersOwner:
    def __init__(self) -> None:
        self.config = {"use_multi_api_keys": False}
        self.env_built = []
        self.compiled = []

    def _build_epub_compile_env(self, folder):
        self.env_built.append(folder)
        os.environ["UPDATE_HTML_HEADERS"] = "1"
        os.environ["SAVE_HEADER_TRANSLATIONS"] = "1"

    def _run_epub_compile(self, folder):
        self.compiled.append(folder)
        path = os.path.join(folder, "book.epub")
        Path(path).write_bytes(b"epub")
        return types.SimpleNamespace(ok=True, path=path, error=None)


def test_headers_adapter_runs_the_shared_entry_then_rebuilds_the_first(tmp_path, monkeypatch):
    """The job runs translate_headers_standalone.run_translate_headers_now (the desktop worker) with
    translate_headers_now as its runner: sources as the selection, the Library folders, the job's Stop."""
    calls = []
    module = types.ModuleType("translate_headers_standalone")

    def translate_headers_now(gui, *, show_error=None, process_events=None, output_dir_for=None):
        found = [(source, output_dir_for(source)) for source in gui.selected_files]
        calls.append({"found": found, "current": gui.get_current_epub_path(), "stop": gui._headers_stop_requested,
                      "client": gui.api_client})
        gui.append_log("📊 Will process 3 EPUB/PDF file(s)")
        show_error("Error", "boom")
        return 2, 1

    def run_translate_headers_now(gui, model, api_key, *, headers_runner=None, rebuild_epub=True):
        calls.append({"model": model, "key": api_key, "rebuild": rebuild_epub})
        gui.api_client = "client"
        try:
            headers_runner(gui)
        finally:
            del gui.api_client

    module.translate_headers_now = translate_headers_now
    module.run_translate_headers_now = run_translate_headers_now
    monkeypatch.setitem(sys.modules, "translate_headers_standalone", module)
    monkeypatch.delenv("UPDATE_HTML_HEADERS", raising=False)
    first = tool_target(tmp_path, "One")
    second = tool_target(tmp_path, "Two")
    no_folder = tool_target(tmp_path, "Three", folder=False)
    spec = hm.headers_spec([first, second, no_folder])
    assert spec.kind == "translate_headers" and spec.params["rebuild_epub"] is True
    owner = HeadersOwner()
    owner.model_var = "gpt-4o"
    owner.api_key_entry = types.SimpleNamespace(text=lambda: " sk-test ")
    ctx = FakeCtx(owner, params=spec.params, inputs=spec.inputs)
    result = headers_kind.run(ctx)
    assert calls[0] == {"model": "gpt-4o", "key": "sk-test", "rebuild": False}  # mobile rebuilds below
    assert calls[1]["found"] == [(first.source, first.folder), (second.source, second.folder), (no_folder.source, None)]
    assert calls[1]["current"] == first.source and calls[1]["stop"] is False and calls[1]["client"] == "client"
    assert not hasattr(owner, "api_client")  # the temporary client lived on the job's view only
    assert owner.env_built == [first.folder]  # the compile env of the first workspace
    assert owner.compiled == [first.folder]  # desktop: the current EPUB is rebuilt
    assert result["ok"] is True and result["outputs"] == [os.path.join(first.folder, "book.epub")]
    assert "📊 Will process 3 EPUB/PDF file(s)" in ctx.logs and "❌ boom" in ctx.logs
    assert ctx.results == {"headers_successful": 2, "headers_failed": 1}
    # stopped: no rebuild, ok None
    owner2 = HeadersOwner()
    ctx2 = FakeCtx(owner2, params=spec.params, inputs=spec.inputs)
    ctx2.stop = True
    assert headers_kind.run(ctx2)["ok"] is None and owner2.compiled == []
    assert calls[-1]["stop"] is True


def test_compile_pdf_of_an_epub_workspace_lists_the_pdf(tmp_path):
    """Compile PDF of an EPUB workspace (the EPUB compile with "Create PDF after EPUB"): the PDF the run
    wrote is a job output next to the EPUB, so the Result card can share / open it."""
    folder = tmp_path / "Output" / "Book"
    folder.mkdir(parents=True)
    old = folder / "old.pdf"
    old.write_bytes(b"%PDF old")
    os.utime(old, (1_000_000, 1_000_000))

    class Owner:
        def _run_epub_compile(self, target):
            Path(target, "Book.pdf").write_bytes(b"%PDF new")
            path = os.path.join(target, "Book.epub")
            Path(path).write_bytes(b"epub")
            return types.SimpleNamespace(ok=True, path=path, error=None)

    spec = cm.compile_spec(tg.ToolTarget("Book", folder=str(folder), kind="epub"), "pdf")
    assert spec.kind == "compile_epub" and spec.params["pdf_after_epub"] is True
    result = compile_kind.run_epub(FakeCtx(Owner(), params=spec.params, inputs=spec.inputs))
    assert result["ok"] is True
    assert result["outputs"] == [str(folder / "Book.epub"), str(folder / "Book.pdf")]  # not the untouched old.pdf
    assert cm.outputs_of(result["outputs"]) == result["outputs"]  # both get Result card rows
    plain =compile_kind.run_epub(FakeCtx(Owner(), params={"folder": str(folder)}))
    assert plain["outputs"] == [str(folder / "Book.epub")]  # Compile EPUB lists only the EPUB


# ==========================================================================
# Converter adapters (validate / rename)
# ==========================================================================


def test_validate_and_rename_kinds(tmp_path, monkeypatch):
    good = tmp_path / "Good"
    bad = tmp_path / "Bad"
    good.mkdir()
    bad.mkdir()
    fake = types.ModuleType("TransateKRtoEN")
    fake.validate_epub_structure = lambda folder: os.path.basename(folder) == "Good"
    fake.check_epub_readiness = lambda folder: os.path.basename(folder) == "Good"
    monkeypatch.setitem(sys.modules, "TransateKRtoEN", fake)
    spec = cm.validate_spec([tg.ToolTarget("Good", folder=str(good)), tg.ToolTarget("Bad", folder=str(bad))])
    ctx = FakeCtx(object(), params=spec.params, inputs=spec.inputs)
    assert compile_kind.run_validate(ctx)["ok"] is True
    assert ctx.results == {"validation": ["✅ Good: All structure files present", "❌ Bad: Missing critical EPUB files"],
                           "all_passed": False}
    assert "  ✅ Good: PASSED" in ctx.logs and "  ❌ Bad: Missing critical files" in ctx.logs
    import output_tools_core

    assert compile_kind.run_validate.__module__ == compile_kind.__name__
    assert "validate_epub_outputs" in compile_kind.run_validate.__code__.co_names
    assert output_tools_core.validate_epub_outputs is not None

    calls = []
    monkeypatch.setitem(sys.modules, "output_naming", types.SimpleNamespace(
        _rename_output_files_for_retain=lambda gui, retain, output_dir=None: calls.append((gui, retain, output_dir))
        or ("renamed", 3)))
    owner = types.SimpleNamespace(config={"retain_source_extension": True})
    rctx = FakeCtx(owner, params=cm.rename_spec(tg.ToolTarget("Good", folder=str(good))).params)
    assert compile_kind.run_rename(rctx)["ok"] is True
    assert calls == [(owner, True, str(good))] and rctx.results["rename"] == "✅ 3 files renamed"
    assert compile_kind.rename_message(("no_opf",)) == "📁 No OPF package found"
    assert compile_kind.rename_message(None) == "📄 No files to rename"


# ==========================================================================
# Models
# ==========================================================================


def _literal_assignment(path: Path, name: str, function: str = None):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if function and isinstance(node, ast.FunctionDef) and node.name != function:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in sub.targets):
                return ast.literal_eval(sub.value)
    raise AssertionError(f"{name} not found in {path.name}")


def test_mode_cards_match_the_desktop_mode_dialog():
    desktop = _literal_assignment(SRC_DIR / "QA_Scanner_GUI.py", "mode_data")
    mine = [{"value": c.value, "emoji": c.emoji, "title": c.title, "subtitle": c.subtitle,
             "features": list(c.features), "recommendation": c.recommendation} for c in qm.MODE_CARDS]
    assert mine == [{k: d[k] for k in ("value", "emoji", "title", "subtitle", "features", "recommendation")}
                    for d in desktop]
    assert qm.DISPLAY_ORDER[0] == "quick-scan" and set(qm.DISPLAY_ORDER) == {c.value for c in qm.MODE_CARDS}


def test_metadata_field_tables_are_the_desktop_dialogs():
    """U7: the tables and rules live in metadata_batch_translator; the desktop dialog uses them."""
    import metadata_batch_translator as mbt

    assert hm.STANDARD_FIELDS is mbt.METADATA_STANDARD_FIELDS
    assert hm.DEFAULT_ENABLED_FIELDS is mbt.METADATA_DEFAULT_ENABLED_FIELDS
    tree = ast.parse((SRC_DIR / "metadata_batch_translator.py").read_text(encoding="utf-8-sig"))
    dialog = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "configure_metadata_fields")
    assigned = {t.id: s.value.id for s in ast.walk(dialog) if isinstance(s, ast.Assign) and isinstance(s.value, ast.Name)
                for t in s.targets if isinstance(t, ast.Name)}
    assert assigned["standard_fields"] == "METADATA_STANDARD_FIELDS"
    assert assigned["default_enabled_fields"] == "METADATA_DEFAULT_ENABLED_FIELDS"
    called = {n.func.id for n in ast.walk(dialog) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert {"saved_metadata_fields_for_epub", "metadata_field_checked", "store_metadata_field_selection",
            "final_metadata_fields_config"} <= called


def test_custom_mode_conversions_follow_the_desktop_dialog(monkeypatch):
    monkeypatch.setitem(sys.modules, "qa_scan_runtime", fake_qa_runtime())
    defaults = qm.custom_defaults()
    assert defaults["similarity"] == 85
    assert qm.custom_values(None, defaults) == defaults
    saved = {"thresholds": {"similarity": 0.7, "semantic": 0.6, "structural": 0.9, "word_overlap": 0.75,
                            "minhash_threshold": 0.8}, "consecutive_chapters": 4, "check_all_pairs": True,
             "sample_size": -1, "min_text_length": 700}
    values = qm.custom_values(saved, defaults)
    assert values["similarity"] == 70 and values["semantic"] == 60 and values["consecutive_chapters"] == 4
    assert qm.custom_saved(values) == saved


def test_report_listing_summary_and_webview_html(tmp_path):
    folder = tmp_path / "Output" / "Book"
    report = Path(qa_kind.report_path_for(str(folder)))
    report.parent.mkdir(parents=True)
    report.write_text("<html><head><title>QA</title></head><body><table><tr><td>"
                      "<a href='../response_0002_ch2.html' target='_blank'>response_0002_ch2.html</a></td></tr>"
                      "</table></body></html>", encoding="utf-8")
    (report.parent / "validation_results.json").write_text(json.dumps([
        {"file_index": 1, "filename": "response_0001_ch1.html", "score": 0, "issues": [], "preview": "ok"},
        {"file_index": 2, "filename": "response_0002_ch2.html", "score": 2,
         "issues": ["korean_text_found_12", "DUPLICATE: exact_or_near_copy_of_x"], "preview": "<p>x</p>",
         "duplicate_confidence": 0.93}]), encoding="utf-8")
    (tmp_path / "Output" / "Other").mkdir()
    entries = qm.list_reports([str(folder)], [str(tmp_path / "Output")])
    assert [e.path for e in entries] == [str(report)] and entries[0].title == "Book"
    assert qm.report_folder(str(report)) == str(folder)
    summary = qm.load_report_summary(str(report))
    assert (summary.total, summary.with_issues, summary.clean) == (2, 1, 1)
    assert summary.rows[0].filename == "response_0002_ch2.html" and summary.rows[0].confidence == 0.93
    assert summary.issue_counts == {"DUPLICATE": 1, "korean": 1}
    page = qm.annotate_report_html(report.read_text(encoding="utf-8"), nonce="N0nce", event_path="/tok/__ev")
    assert "href='../" not in page and 'data-glqa-file="response_0002_ch2.html"' in page
    assert "Open in Chapters" in page and '<script nonce="N0nce">' in page and '"/tok/__ev"' in page
    assert "<meta name='viewport'" in page.split("</head>")[0]
    event = qm.parse_report_event('GLQA:{"type":"open","file":"../../x/response_0002_ch2.html","seq":3,'
                                  '"action":"chapters"}')
    assert event == {"type": "open", "file": "response_0002_ch2.html", "seq": 3, "action": "chapters"}
    assert qm.parse_report_event("noise") is None and qm.parse_report_event({"type": "evil"}) is None


def test_artifact_delete_plan_and_recycled_link(tmp_path):
    if not _has("translation_artifacts"):
        pytest.skip("translation_artifacts not importable")
    import translation_artifacts as ta

    folder = tmp_path / "Output" / "Book"
    folder.mkdir(parents=True)
    (folder / "translated_headers.txt").write_text("headers", encoding="utf-8")
    (folder / "TOC.txt").write_text("toc", encoding="utf-8")
    progress = {"chapters": {
        "__translation_artifact__:headers": {"status": "completed", "model_name": "RECYCLED",
                                             "output_file": "translated_headers.txt"},
        "__translation_artifact__:toc": {"status": "completed", "model_name": "gpt-x", "output_file": "TOC.txt"}}}
    (folder / "translation_progress.json").write_text(json.dumps(progress), encoding="utf-8")
    empty = tmp_path / "Output" / "Empty"
    empty.mkdir()
    targets = [tg.ToolTarget("Book", folder=str(folder)), tg.ToolTarget("Empty", folder=str(empty)),
               tg.ToolTarget("Gone", folder="")]
    plan = hm.plan_artifact_delete(targets, "headers")
    assert plan.found == [("Book", str(folder / "translated_headers.txt"))] and plan.has_linked
    assert plan.not_found == [("Empty", "translated_headers.txt not found"), ("Gone", "No output directory found")]
    text = plan.question_text()
    assert text.startswith("Summary for 3 EPUB file(s):") and "This will allow headers to be re-translated" in text
    assert "RECYCLED link detected" in text and "delete only the header file?" in text
    message, ok = hm.execute_artifact_delete(plan, delete_linked=True)
    assert ok and message.startswith("Successfully deleted 2 file(s):")
    assert not (folder / "translated_headers.txt").exists() and not (folder / "TOC.txt").exists()
    saved = ta.load_translation_artifact_progress(str(folder))
    for kind in ("headers", "toc"):
        _key, entry = ta.translation_artifact_progress_entry(saved, kind)
        assert entry["status"] == "pending" and not entry.get("model_name")
    toc_plan = hm.plan_artifact_delete([targets[0]], "toc")
    assert toc_plan.found == [] and toc_plan.not_found == [("Book", "TOC.txt not found")]
    assert hm.execute_artifact_delete(toc_plan) == ("No files were successfully deleted.", False)


def test_metadata_fields_detection_and_save_rules(tmp_path):
    if not _has("metadata_batch_translator"):
        pytest.skip("metadata_batch_translator not importable")
    epub = make_epub(tmp_path / "raw" / "Meta.epub", extra_meta='<meta name="calibre:series" content="Saga"/>')
    fields = hm.detect_fields(epub)
    assert fields["title"] == "Raw Book" and fields["creator"] == "Author" and fields["series"] == "Saga"
    sync = {}
    checks = hm.initial_checks(fields, {}, sync)
    assert checks["title"] and checks["subject"] and not checks["creator"]
    assert hm.initial_checks({"publisher": "x"}, {"publisher": True}, sync)["publisher"] is True
    assert hm.initial_checks({"creator": "x"}, {"creator": True}, sync)["creator"] is False  # synced (1st EPUB)
    assert hm.initial_checks({"title": "x"}, {"title": False}, sync)["title"] is True  # synced from the 1st EPUB
    assert hm.fields_config({"x": True}, {epub: {"title": True, "creator": False}}, [epub]) == {
        "title": True, "creator": False}
    other = str(tmp_path / "raw" / "Other.epub")
    merged = hm.fields_config({"_per_epub": {"Old.epub": {"rights": True}}},
                              {epub: {"title": True}, other: {"subject": False}}, [epub, other])
    assert merged["_per_epub"] == {"Old.epub": {"rights": True}, "Meta.epub": {"title": True},
                                   "Other.epub": {"subject": False}}
    assert merged["title"] and merged["rights"] and merged["subject"] is False
    assert hm.saved_selection(merged, other) == {"subject": False}
    assert hm.saved_selection({"title": False}, other) == {"title": False}
    folder = tmp_path / "Output" / "Meta"
    folder.mkdir(parents=True)
    assert hm.existing_metadata_warning([str(folder)]) is None
    (folder / "metadata.json").write_text("{}", encoding="utf-8")
    assert hm.existing_metadata_warning([str(folder)]).startswith("metadata.json already exists for this EPUB.")
    assert "already exists for 1 of the 2 selected EPUBs" in hm.existing_metadata_warning([str(folder), ""])


def test_compile_specs_fonts_and_css(tmp_path):
    epub_ws = tg.ToolTarget("Book", folder=str(tmp_path / "Book"), kind="epub", bid="aaaaaaaaaaaa")
    pdf_ws = tg.ToolTarget("Scan", folder=str(tmp_path / "Scan"), kind="pdf")
    spec = cm.compile_spec(epub_ws, "pdf")
    assert spec.kind == "compile_epub" and spec.params["config_overrides"] == {"enable_pdf_output": True}
    assert spec.origin == {"type": "library", "bid": "aaaaaaaaaaaa", "label": "Library · Book"}
    assert cm.compile_spec(pdf_ws, "pdf").kind == "compile_pdf"
    assert cm.compile_spec(epub_ws, "epub").params == {"folder": epub_ws.folder}
    with pytest.raises(ValueError, match="PDF workspace"):
        cm.compile_spec(pdf_ws, "epub")
    assert cm.rename_spec(epub_ws, True).params == {"folder": epub_ws.folder, "retain": True}
    fonts = tmp_path / "fonts"
    zipped = tmp_path / "pack.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.writestr("sub/A.ttf", b"a")
        zf.writestr("readme.txt", b"r")
    single = tmp_path / "B.otf"
    single.write_bytes(b"b")
    (tmp_path / "skip.txt").write_text("x")
    assert cm.import_fonts([str(zipped), str(single), str(tmp_path / "skip.txt")], str(fonts)) == 2
    assert sorted(os.listdir(fonts)) == ["A.ttf", "B.otf"] and cm.count_fonts(str(fonts)) == 2
    assert cm.clear_fonts(str(fonts)) == 2 and cm.count_fonts(str(fonts)) == 0
    css = tmp_path / "style.css"
    css.write_text("p{}")
    stored = cm.import_css(str(css), str(tmp_path / "imports"))
    assert stored == str(tmp_path / "imports" / "style.css") and os.path.isfile(stored)
    with pytest.raises(ValueError):
        cm.import_css(str(single), str(tmp_path / "imports"))


def test_targets_from_the_library_snapshot_and_browse(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "qa_scan_runtime", fake_qa_runtime())
    raw = make_epub(tmp_path / "Library" / "Raw" / "Book.epub")
    out = tmp_path / "Output" / "Book"
    out.mkdir(parents=True)
    (out / "response_ch001.html").write_text("x", encoding="utf-8")
    chat = tmp_path / "Output" / "Direct Text" / "Chat 1" / "att"
    chat.mkdir(parents=True)
    (chat / "translation_progress.json").write_text("{}", encoding="utf-8")
    book = {"name": "Book", "output_folder": str(out), "raw_source_path": raw, "type": "in_progress", "mtime": 5.0}
    service = types.SimpleNamespace(
        snapshot=types.SimpleNamespace(all_books=lambda: [book, {"name": "Nothing"}], in_progress=[book]),
        raw_source=lambda b: b.get("raw_source_path") or "", bid_for=lambda b: "bbbbbbbbbbbb")
    rows = tg.library_targets(service)
    assert len(rows) == 1 and rows[0].folder == str(out) and rows[0].source == raw and rows[0].bid == "bbbbbbbbbbbb"
    assert tg.recent_output_targets(service)[0].key == rows[0].key
    chats = tg.chat_workspace_targets(str(tmp_path / "Output" / "Direct Text"))
    assert [c.folder for c in chats] == [str(chat)] and chats[0].direct_text
    picked = tg.target_for_source(raw, output_root=str(tmp_path / "Output"))
    assert picked.folder == str(out) and picked.origin == "browse"
    assert tg.target_for_source(raw, candidates=lambda *a, **k: []).folder == ""


# ==========================================================================
# Screens (fake Flet session)
# ==========================================================================


class FakeJobs:
    """JobsFeature surface the tool screens use (submit, has_kind, snapshot, on_transition, request_stop)."""

    def __init__(self) -> None:
        self.specs: list = []
        self.listeners: list = []
        self.stops: list = []
        self.snaps: dict = {}

    def has_kind(self, kind):
        return kind in job_kinds.KIND_MODULES

    async def submit(self, spec):
        from glossarion_mobile.services.jobs import JobSnapshot

        self.specs.append(spec)
        job_id = f"job{len(self.specs)}"
        self.snaps[job_id] = JobSnapshot(id=job_id, spec=spec, state=JobState.QUEUED, created=1.0)
        return job_id

    def snapshot(self, job_id=None):
        return self.snaps.get(job_id)

    def view(self):
        """The JobsView shape: the running job and the queued ones (jobs submitted here start QUEUED)."""
        live = [snap for snap in self.snaps.values() if not snap.is_terminal]
        active = next((snap for snap in live if snap.state != JobState.QUEUED), None)
        return types.SimpleNamespace(active=active, queue=tuple(snap for snap in live if snap is not active))

    def on_transition(self, callback):
        self.listeners.append(callback)
        return lambda: self.listeners.remove(callback) if callback in self.listeners else None

    def request_stop(self, job_id=None, **kwargs):
        self.stops.append(job_id)
        return "graceful"

    def run(self, job_id):
        import dataclasses

        snap = dataclasses.replace(self.snaps[job_id], state=JobState.RUNNING, started=2.0, phase="Scanning")
        self.snaps[job_id] = snap
        for callback in list(self.listeners):
            callback(snap, JobState.QUEUED)
        return snap

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
        self.added: list = []

    async def pick_files(self, **kwargs):
        paths = self.picks.pop(0) if self.picks else []
        return [types.SimpleNamespace(path=p, name=os.path.basename(p)) for p in paths]

    async def share(self, paths, **kwargs):
        self.shared.append(list(paths))
        return True

    def export_options(self, path):
        from glossarion_mobile.services.files import ExportOption

        return [ExportOption("share", "Share…", "IOS_SHARE"), ExportOption("save", "Save to…", "SAVE_ALT")]

    def add_to_library(self, path, translated=False):
        self.added.append((path, translated))
        return types.SimpleNamespace(name=os.path.basename(path), path=path)


def _tb():
    return _load("_glossarion_tools_tb_helpers", "test_bootstrap.py")


def _ctx(page, *, service=None, store=None, jobs=None, files=None, prefs=None, **kwargs):
    from glossarion_mobile.ui.tools.common import ToolsContext

    navigated: list = []
    notes: list = []
    ctx = ToolsContext(service=service, page=page,
                       navigate=lambda name, params=None, query=None: navigated.append((name, params, query)),
                       notify=lambda message, action=None, on_action=None: notes.append(message),
                       jobs=jobs if jobs is not None else FakeJobs(), files=files, prefs=prefs or FakePrefs(),
                       platform="android", store=store if store is not None else {}, **kwargs)
    ctx.navigated, ctx.notes = navigated, notes
    ctx.extras["answers"] = []
    return ctx


def _mount(page, body):
    page.views[0].controls.append(body)
    page.update()


async def _settle(times=5):
    for _ in range(times):
        await asyncio.sleep(0.01)


@needs_flet
def test_tools_hub_tiles_and_availability(tmp_path):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.feature import IMPLEMENTED_ROUTES
    from glossarion_mobile.ui.tools.hub import HUB_GROUPS, ToolsHubScreen, tile_available

    tb = _tb()

    async def scenario():
        _conn, session = tb._fake_session("android")
        page = session.page
        ctx = _ctx(page)
        ctx.prefs.set("tools_last_sources", {"tools.qa": "My Novel"})
        screen = ToolsHubScreen(parse_route("/tools"), ctx, implemented=IMPLEMENTED_ROUTES | {"tools.progress"})
        _mount(page, screen.get_body())
        names = [t.name for _g, tiles in HUB_GROUPS for t in tiles]
        assert names[:4] == ["Async batch", "Review generator", "Headers & metadata", "RPG Maker"]
        assert tile_available(next(t for _g, ts in HUB_GROUPS for t in ts if t.key == "tools.manga")) is None  # U8
        assert tile_available(next(t for _g, ts in HUB_GROUPS for t in ts if t.key == "tools.async")) is None  # U7
        assert tile_available(next(t for _g, ts in HUB_GROUPS for t in ts if t.key == "tools.qa"),
                              IMPLEMENTED_ROUTES) is None
        assert screen.tiles["tools.manga"].on_click is not None  # U8
        assert screen.tiles["tools.qa"].content.controls[1].controls[1].value == "My Novel"
        screen.tiles["tools.convert.validate"].on_click(None)
        assert ctx.navigated[-1] == ("tools.convert", None, {"tab": "validate"})
        screen.tiles["tools.qa"].on_click(None)
        assert ctx.navigated[-1] == ("tools.qa", None, None)

    asyncio.run(scenario())


@needs_flet
def test_qa_screen_modes_sources_prechecks_and_job(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "qa_scan_runtime", fake_qa_runtime())
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.qa_screen import QaScannerScreen, qa_eligibility

    tb = _tb()
    target = tool_target(tmp_path, "Alpha")
    no_source = tool_target(tmp_path, "Beta", source=False)

    async def scenario():
        _conn, session = tb._fake_session("android")
        page = session.page
        store = {"qa_scanner_settings": {"quick_scan_sample_size": 1000}, "qa_auto_search_output": True}
        ctx = _ctx(page, store=store)
        screen = QaScannerScreen(parse_route("/tools/qa"), ctx)
        _mount(page, screen.get_body())
        screen.did_show()
        await _settle()
        assert screen.mode == "quick-scan" and list(screen.mode_cards) == list(qm.DISPLAY_ORDER)
        assert screen.start_button.disabled and screen.run_status.value == "Choose a folder to scan"
        screen.select_mode("aggressive")
        assert ctx.tool_state["qa"]["mode"] == "aggressive"
        screen.sample_field.value = "-1"
        screen._on_sample()
        assert store["qa_scanner_settings"]["quick_scan_sample_size"] == -1
        screen.auto_search.value = False
        screen._on_auto_search()
        assert store["qa_auto_search_output"] is False
        assert qa_eligibility(tg.ToolTarget("chat", folder=str(tmp_path), direct_text=True)) is not None
        # word count on + no source anywhere -> the desktop question; "continue" scans without it
        screen.set_targets([no_source])
        page.update()
        assert not screen.start_button.disabled
        screen.sample_field.value = "abc"  # not a number: Start refuses (the field shows the error)
        assert await screen.start() is None and ctx.notes[-1] == "Fix the Quick Scan sample size first"
        assert screen.sample_field.error and not ctx.jobs.specs
        screen.sample_field.value = "250"  # typed, never blurred (a tap on Start keeps the focus on phones)
        ctx.extras["answers"] = ["continue"]
        job_id = await screen.start()
        assert store["qa_scanner_settings"]["quick_scan_sample_size"] == 250
        assert ctx.extras["asked"][0][0] == qm.NO_SOURCE_TITLE
        spec = ctx.jobs.specs[-1]
        assert spec.kind == "qa_scan" and spec.params["mode"] == "aggressive" and spec.params["disable_word_count"]
        assert spec.params["targets"] == [{"folder": no_source.folder, "source": None}]
        assert ctx.prefs.get("tools_last_sources")["tools.qa"] == "Beta"
        page.update()
        # mismatch question for a single folder whose source name differs; cancel keeps everything idle
        mismatched = target.with_source(make_epub(tmp_path / "raw" / "Other.epub"))
        screen.set_targets([mismatched])
        ctx.extras["answers"] = ["cancel"]
        assert await screen.start() is None and ctx.extras["asked"][-1][0] == qm.MISMATCH_TITLE
        assert len(ctx.jobs.specs) == 1
        # the job ends -> the report list reloads and the run line shows the outcome
        report = qa_kind.report_path_for(no_source.folder)
        os.makedirs(os.path.dirname(report), exist_ok=True)
        Path(report).write_text("<html></html>", encoding="utf-8")
        ctx.output_root = str(tmp_path / "Output")
        ctx.jobs.finish(job_id, outputs=[report], result={"qa_reports": [report]})
        await _settle(10)
        page.update()
        assert screen.run_status.value.startswith("Done") and report in screen.report_rows
        rid = screen.open_report(report)
        assert ctx.navigated[-1] == ("tools.qa.report", {"rid": rid}, None)
        assert ctx.prefs.resolve_file_ref(rid) == os.path.abspath(report)
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_qa_screen_shows_the_mobile_sample_size_and_migrates_a_saved_1000_once(tmp_path, monkeypatch):
    """Owner device report #7 (2026-10-08): the sample size field shows 0 (duplicate check off), not the
    desktop 1000; a 1000 an earlier build saved becomes 0 once; the field stays editable."""
    monkeypatch.setitem(sys.modules, "qa_scan_runtime", fake_qa_runtime())
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.qa_screen import QaScannerScreen, open_qa_report

    tb = _tb()
    target = tool_target(tmp_path, "Alpha")

    async def scenario():
        _conn, session = tb._fake_session("android")
        page = session.page
        # nothing saved: 0 shown; Start leaves the untouched default unsaved (the job applies it)
        fresh: dict = {"qa_scanner_settings": {"check_word_count_ratio": False}}
        ctx = _ctx(page, store=fresh)
        screen = QaScannerScreen(parse_route("/tools/qa"), ctx)
        _mount(page, screen.get_body())
        assert screen.sample_field.value == "0" and screen.mode == "quick-scan"
        assert ctx.prefs.get(qm.QUICK_SAMPLE_MIGRATION_PREF) is True
        screen.set_targets([target])
        assert await screen.start() and fresh["qa_scanner_settings"] == {"check_word_count_ratio": False}
        assert ctx.jobs.specs[-1].params == {"mode": "quick-scan", "targets": [target.to_param()]}
        # a config an earlier build saved with the desktop default: 0 after the one-time migration
        saved = {"qa_scanner_settings": {"quick_scan_sample_size": 1000, "check_word_count_ratio": False}}
        _conn2, session2 = tb._fake_session("android")  # a fresh page per screen (same control keys)
        ctx2 = _ctx(session2.page, store=saved)
        screen2 = QaScannerScreen(parse_route("/tools/qa"), ctx2)
        _mount(session2.page, screen2.get_body())
        assert screen2.sample_field.value == "0"
        assert saved["qa_scanner_settings"] == {"quick_scan_sample_size": 0, "check_word_count_ratio": False}
        # still editable, and the migration never runs again: a 1000 typed now stays
        screen2.sample_field.value = "1000"
        assert screen2._on_sample() and saved["qa_scanner_settings"]["quick_scan_sample_size"] == 1000
        _conn3, session3 = tb._fake_session("android")
        ctx2.page = session3.page
        screen3 = QaScannerScreen(parse_route("/tools/qa"), ctx2)
        _mount(session3.page, screen3.get_body())
        assert screen3.sample_field.value == "1000" and saved["qa_scanner_settings"]["quick_scan_sample_size"] == 1000
        # the report opener the chat's QA card shares with the screen
        report = qa_kind.report_path_for(target.folder)
        rid = open_qa_report(ctx2, report)
        assert ctx2.navigated[-1] == ("tools.qa.report", {"rid": rid}, None)
        assert ctx2.prefs.resolve_file_ref(rid) == os.path.abspath(report)
        assert open_qa_report(ctx2, "") is None and ctx2.notes[-1] == "Reports cannot be opened in this session"
        # the ChatView shape: navigate / notify, Prefs on its env
        went, said = [], []
        chat_view = types.SimpleNamespace(env=types.SimpleNamespace(prefs=FakePrefs()), notify=said.append,
                                          navigate=lambda name, params=None: went.append((name, params)))
        rid = open_qa_report(chat_view, report)
        assert went == [("tools.qa.report", {"rid": rid})] and chat_view.env.prefs.resolve_file_ref(rid)
        assert open_qa_report(types.SimpleNamespace(env=None, notify=said.append,
                                                    navigate=lambda *a: None), report) is None
        assert said == ["Reports cannot be opened in this session"]

    asyncio.run(scenario())


@needs_flet
def test_qa_custom_mode_saves_first_then_scans_and_row_actions(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "qa_scan_runtime", fake_qa_runtime())
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.qa_screen import QaScannerScreen

    tb = _tb()
    target = tool_target(tmp_path, "Custom")

    async def scenario():
        _conn, session = tb._fake_session("android")
        page = session.page
        ctx = _ctx(page, store={"qa_scanner_settings": {"check_word_count_ratio": False}})
        screen = QaScannerScreen(parse_route("/tools/qa"), ctx)
        _mount(page, screen.get_body())
        screen.set_targets([target])
        screen.select_mode("custom")  # opens the Custom sheet
        assert screen.custom_sheet is not None
        assert await screen.start() is None and not ctx.jobs.specs  # nothing saved yet: the sheet comes first
        sheet = screen.custom_sheet
        sheet.save()
        await _settle(10)
        assert ctx.jobs.specs and ctx.jobs.specs[-1].params["mode"] == "custom"
        assert ctx.store["qa_scanner_settings"]["custom_mode_settings"]["thresholds"]["similarity"] == 0.85
        menu = screen.target_actions(target)
        menu.item("Scan without a source").on_select()
        assert screen.targets[0].source == "" and screen.targets[0].folder == target.folder
        assert menu.item("Remove").destructive

    asyncio.run(scenario())


@needs_flet
def test_qa_custom_sheet_saves_the_desktop_shape(monkeypatch):
    monkeypatch.setitem(sys.modules, "qa_scan_runtime", fake_qa_runtime())
    from glossarion_mobile.ui.tools.qa_custom import CustomModeSheet

    tb = _tb()

    async def scenario():
        _conn, session = tb._fake_session("android")
        ctx = _ctx(session.page, store={})
        saved = []
        sheet = CustomModeSheet(ctx, on_saved=saved.append).show(session.page)
        session.page.update()
        sheet.sliders["similarity"].value = 70
        sheet.consecutive.value = "99"  # clamped to 10
        sheet.sample.value = "-1"
        sheet.check_all.value = True
        value = sheet.save()
        assert value["thresholds"]["similarity"] == 0.7 and value["consecutive_chapters"] == 10
        assert value["sample_size"] == -1 and value["check_all_pairs"] is True
        assert ctx.store["qa_scanner_settings"]["custom_mode_settings"] == value
        await _settle()
        assert saved == [value]
        again = CustomModeSheet(ctx)
        assert again.values["similarity"] == 70 and again.values["consecutive_chapters"] == 10
        again.reset()
        assert again.collect()["similarity"] == 85

    asyncio.run(scenario())


@needs_flet
def test_qa_report_viewer_native_fallback_and_open_in_chapters(tmp_path):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.qa_report import QaReportScreen

    tb = _tb()
    folder = tmp_path / "Output" / "Book"
    report = Path(qa_kind.report_path_for(str(folder)))
    report.parent.mkdir(parents=True)
    report.write_text("<html><body><h1>Translation QA Report</h1></body></html>", encoding="utf-8")
    (report.parent / "validation_results.json").write_text(json.dumps([
        {"file_index": 2, "filename": "response_0002_ch2.html", "score": 2, "issues": ["korean_text_found_3"],
         "preview": "x"}]), encoding="utf-8")

    async def scenario():
        _conn, session = tb._fake_session("windows")
        page = session.page
        prefs = FakePrefs()
        rid = prefs.file_ref(str(report), kind="qa_report")
        book = {"name": "Book", "output_folder": str(folder), "type": "in_progress"}
        service = types.SimpleNamespace(snapshot=types.SimpleNamespace(all_books=lambda: [book]),
                                        bid_for=lambda b: "cccccccccccc")
        ctx = _ctx(page, service=service, prefs=prefs, open_url=lambda url: url)
        screen = QaReportScreen(parse_route(f"/tools/qa/report/{rid}"), ctx)
        assert screen.folder == str(folder)
        _mount(page, screen.get_body())
        assert await screen.load() == "native"
        page.update()
        assert screen.headline.value.startswith("Total Files Scanned: 1 · Files with Issues: 1")
        assert screen.browser_button.visible
        screen.handle_event({"type": "open", "file": "response_0002_ch2.html", "seq": 1})
        screen.handle_event({"type": "open", "file": "response_0002_ch2.html", "seq": 1})  # duplicate channel
        assert ctx.navigated == [("library.book", {"bid": "cccccccccccc"}, {"tab": "chapters", "filter": "failed"})]
        missing = QaReportScreen(parse_route("/tools/qa/report/0123456789ab"), ctx)
        assert missing.path == "" and missing.get_body().title == "Report not found"

    asyncio.run(scenario())


@needs_flet
@pytest.mark.skipif(importlib.util.find_spec("flet_webview") is None, reason="flet_webview not installed")
def test_qa_report_webview_falls_back_when_the_page_never_reports_in(tmp_path):
    """Android / iOS: a WebView whose page never posts "ready" (blocked cleartext, a broken system WebView,
    a resource error) is replaced by the native report, like the Reader; a page that reported in stays."""
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.qa_report import QaReportScreen

    tb = _tb()
    folder = tmp_path / "Output" / "Book"
    report = Path(qa_kind.report_path_for(str(folder)))
    report.parent.mkdir(parents=True)
    report.write_text("<html><head></head><body><h1>Translation QA Report</h1></body></html>", encoding="utf-8")
    # no validation_results.json: with one the parsed native report is the view (U13 item 3)

    async def scenario():
        _conn, session = tb._fake_session("android")
        page = session.page
        prefs = FakePrefs()
        rid = prefs.file_ref(str(report), kind="qa_report")
        ctx = _ctx(page, prefs=prefs, webview_ok=lambda: True)
        screens = []
        for heard in (False, True):
            screen = QaReportScreen(parse_route(f"/tools/qa/report/{rid}"), ctx)
            screens.append(screen)
            _mount(page, screen.get_body())
            assert await screen.load() == "webview" and screen.server is not None
            if heard:
                screen.handle_event({"type": "ready", "seq": 1})
            screen._on_web_error(types.SimpleNamespace(data="net::ERR_CLEARTEXT_NOT_PERMITTED"))
            assert await screen.check_webview(0) is (not heard)
            assert screen.renderer == ("webview" if heard else "native")
            if not heard:
                assert screen.server is None and screen.webview is None
                assert ctx.notes[-1] == "The report page view is unavailable here; showing the summary"
        for screen in screens:
            screen.dispose()

    asyncio.run(scenario())


def test_report_webview_page_through_the_reader_server(tmp_path):
    """The served report runs only the nonce'd viewer script and posts events back."""
    import urllib.request

    from glossarion_mobile.services.reader_server import ReaderServer

    events = []
    server = ReaderServer(on_event=events.append)
    server.start()
    try:
        html = qm.annotate_report_html("<html><head></head><body><a href='../a.html' target='_blank'>a.html</a>"
                                       "</body></html>", nonce="abcDEF123", event_path=server.event_path)
        url = server.publish(html, name="report.html", script_nonce="abcDEF123")
        with urllib.request.urlopen(url, timeout=5) as response:
            csp = response.headers["Content-Security-Policy"]
            body = response.read().decode("utf-8")
        assert "script-src 'nonce-abcDEF123'" in csp and 'data-glqa-file="a.html"' in body
        request = urllib.request.Request(server.origin + server.event_path, method="POST",
                                         data=json.dumps({"type": "open", "file": "a.html", "seq": 1}).encode(),
                                         headers={"Content-Type": "application/json"})
        urllib.request.urlopen(request, timeout=5).close()
        assert qm.parse_report_event(events[0]) == {"type": "open", "file": "a.html", "seq": 1, "action": "file",
                                                    "_channel": "http"}
    finally:
        server.stop()


@needs_flet
def test_converter_screen_compile_validate_rename_and_options(tmp_path):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.converter import ConverterScreen

    tb = _tb()
    target = tool_target(tmp_path, "Conv")
    css = tmp_path / "pick" / "style.css"
    css.parent.mkdir()
    css.write_text("p{}")

    async def scenario():
        _conn, session = tb._fake_session("android")
        page = session.page
        store = {}
        files = FakeFiles([[str(css)]])
        ctx = _ctx(page, store=store, files=files, import_dir=str(tmp_path / "imports"))
        screen = ConverterScreen(parse_route("/tools/convert?tab=validate"), ctx)
        _mount(page, screen.get_body())
        screen.did_show()
        await _settle()
        assert screen.actions_row.controls[0].controls[0].disabled  # no folder yet (button + ReasonChip row)
        screen.set_target(target)
        page.update()
        job_pdf = await screen.compile("pdf")
        spec = ctx.jobs.specs[-1]
        assert spec.kind == "compile_epub" and spec.params["config_overrides"] == {"enable_pdf_output": True}
        epub = os.path.join(target.folder, "Conv.epub")
        pdf = os.path.join(target.folder, "Conv.pdf")
        Path(epub).write_bytes(b"e")
        Path(pdf).write_bytes(b"p")
        ctx.jobs.finish(job_pdf, outputs=[epub, pdf])
        page.update()
        assert screen.result_card.visible and len(screen.result_column.controls) == 2
        sheet = screen.output_actions(pdf)
        assert sheet.item("Open in Reader").disabled_reason == "EPUB and TXT files only"
        assert sheet.item("Add to Library").disabled_reason
        # devfix 2026-10-08 (owner #2): the Reader opens TXT books, so a compiled _translated.txt opens there too
        txt = os.path.join(target.folder, "Conv_translated.txt")
        Path(txt).write_text("Chapter 1\n\ntext", encoding="utf-8")
        txt_sheet = screen.output_actions(txt)
        assert txt_sheet.item("Open in Reader").disabled_reason is None
        assert txt_sheet.item("Add to Library").disabled_reason  # the Completed shelf stays EPUB-only
        assert screen.output_actions(epub).item("Open in Reader").disabled_reason is None
        job_txt = await screen.compile("epub")
        ctx.jobs.finish(job_txt, outputs=[txt])
        page.update()
        assert [c.title.value for c in screen.result_column.controls] == ["Conv_translated.txt"]
        await screen.add_to_library(epub)
        assert files.added == [(epub, True)]
        job_v = await screen.validate()
        assert ctx.jobs.specs[-1].kind == "validate_epub"
        ctx.jobs.finish(job_v, result={"validation": ["✅ Conv: All structure files present"], "all_passed": True})
        page.update()
        assert screen.result_column.controls[0].value == "✅ All Valid!"
        store["retain_source_extension"] = True
        await screen.rename()
        assert ctx.jobs.specs[-1].kind == "rename_outputs" and ctx.jobs.specs[-1].params["retain"] is True
        screen.set_layout("epub2")
        assert store["epub_layout_mode"] == "epub2"
        stored = await screen.load_css()
        assert stored == str(tmp_path / "imports" / "style.css") and store["epub_css_override_path"] == stored
        screen.clear_css()
        assert store["epub_css_override_path"] == ""
        # a PDF workspace: Compile EPUB is refused, Compile PDF runs the PDF workspace compiler
        screen.set_target(tg.ToolTarget("Scan", folder=target.folder, kind="pdf"))
        assert await screen.compile("epub") is None and ctx.notes[-1] == "A PDF workspace compiles to PDF"
        await screen.compile("pdf")
        assert ctx.jobs.specs[-1].kind == "compile_pdf"
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_headers_screen_delete_flow_metadata_and_fields_sheet(tmp_path, monkeypatch):
    if not (_has("translation_artifacts") and _has("metadata_batch_translator")):
        pytest.skip("shared metadata modules not importable")
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.headers_screen import HeadersScreen, MetadataFieldsSheet

    tb = _tb()
    target = tool_target(tmp_path, "Head")
    Path(target.folder, "translated_headers.txt").write_text("h", encoding="utf-8")
    Path(target.folder, "metadata.json").write_text("{}", encoding="utf-8")

    async def scenario():
        _conn, session = tb._fake_session("android")
        page = session.page
        store = {"metadata_translation_mode": "together"}
        ctx = _ctx(page, store=store)
        screen = HeadersScreen(parse_route("/tools/headers"), ctx)
        _mount(page, screen.get_body())
        screen.did_show()
        screen.set_targets([target])
        await _settle(10)
        page.update()
        assert screen.status[target.key]["headers"] and screen.status[target.key]["metadata"]
        # Delete Header Files: summary question (no RECYCLED link) -> Yes -> result
        ctx.extras["answers"] = ["yes", "ok"]
        message, ok = await screen.delete("headers")
        assert ok and not os.path.exists(os.path.join(target.folder, "translated_headers.txt"))
        assert ctx.extras["asked"][0][0] == "Confirm Deletion" and ctx.extras["asked"][1][0] == "Success"
        # translate headers -> job with the targets and the rebuild switch
        job = await screen.translate_headers()
        spec = ctx.jobs.specs[-1]
        assert spec.kind == "translate_headers" and spec.params["targets"] == [{"source": target.source,
                                                                                "folder": target.folder}]
        page.update()
        assert screen.stop_button.visible is True  # a queued job can be stopped (cancelled) too
        assert screen.header_actions.controls[0].controls[0].disabled  # no second header job meanwhile
        screen.active_job = ctx.jobs.snapshot(job)
        screen._on_stop()
        assert ctx.jobs.stops == [job]
        # metadata: "Metadata Already Exists" first; cancel starts nothing, yes queues the job
        ctx.extras["answers"] = ["cancel"]
        assert await screen.translate_metadata() is None
        ctx.extras["answers"] = ["yes"]
        await screen.translate_metadata()
        meta = ctx.jobs.specs[-1]
        assert meta.kind == "metadata" and meta.inputs == (target.source,)
        assert meta.params["output_roots"] == {target.source: os.path.dirname(target.folder)}
        screen.mode_radio.value = "parallel"
        screen._on_mode()
        assert store["metadata_translation_mode"] == "parallel"
        sheet = MetadataFieldsSheet(ctx, [target]).show(page)
        await sheet.load(target.source)
        page.update()
        assert set(sheet.checkboxes) >= {"title", "creator", "subject"}
        assert sheet.checkboxes["title"].value is True and sheet.checkboxes["creator"].value is False
        sheet.checkboxes["creator"].value = True
        sheet._on_toggle("creator")
        saved = sheet.save()
        assert saved["creator"] is True and "_per_epub" not in saved
        assert store["translate_metadata_fields"] == saved
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_tools_feature_wraps_the_screen_factory(tmp_path):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.feature import IMPLEMENTED_ROUTES, ToolsFeature

    tb = _tb()

    async def scenario():
        _conn, session = tb._fake_session("windows")
        fallback = []
        shell = types.SimpleNamespace(screen_factory=lambda match: fallback.append(match.name) or "fallback",
                                      tablet=False)
        app = types.SimpleNamespace(page=session.page, dispatcher=None, shell=shell, library=None, files=None,
                                    jobs=FakeJobs(), prefs=FakePrefs(), haptics=None, intents=None,
                                    navigate_to=lambda *a, **k: None, notify=lambda *a, **k: None,
                                    paths=types.SimpleNamespace(data=str(tmp_path / "data"),
                                                                output=str(tmp_path / "Output")),
                                    settings=None, config_store={}, url_launcher=None)
        feature = await ToolsFeature.install(app)
        assert app.tools is feature and shell.screen_factory == feature.screen_factory
        for route in ("/tools", "/tools/qa", "/tools/convert", "/tools/headers", "/tools/qa/report/0123456789ab"):
            screen = shell.screen_factory(parse_route(route))
            assert screen != "fallback" and screen.get_body() is not None
        assert shell.screen_factory(parse_route("/tools/progress")) == "fallback" and fallback == ["tools.progress"]
        ctx = feature.context()
        assert ctx.chats_root == os.path.join(str(tmp_path / "Output"), "Direct Text")
        assert ctx.import_dir == os.path.join(str(tmp_path / "data"), "imports") and not ctx.webview_ok()
        assert IMPLEMENTED_ROUTES == {"tools", "tools.qa", "tools.qa.report", "tools.convert", "tools.headers",
                                      # U7 (screens: tests_host/test_tools_u7.py)
                                      "tools.async", "tools.review", "tools.sdlxliff", "tools.rpgmaker"}

    asyncio.run(scenario())


@needs_flet
def test_open_in_chapters_route_selects_the_failed_chip(tmp_path, monkeypatch):
    """The QA report's link lands on the Book page Chapters tab with the Failed (QA failed) chip on."""
    tl = _load("_glossarion_tools_library_helpers", "test_library_ui.py")
    lc = tl._core("library_core", "install_library_env", "scan_library", "LibraryShelf")
    if lc is None:
        pytest.skip("library_core (U5 API) not importable")
    fixture = tl.make_workspace(tmp_path)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(fixture["library"]))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(fixture["output"]))
    from glossarion_mobile.services.library import LibraryService
    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.router import build_route, parse_route

    paths = types.SimpleNamespace(library=fixture["library"], output=fixture["output"], cache=fixture["cache"])
    service = LibraryService(paths=paths, config={}, prefs=tl.FakePrefs())
    service.ensure_env()
    tb = _tb()
    try:
        async def scenario():
            await service.refresh()
            book = service.snapshot.in_progress[0]
            bid = service.bid_for(book)
            route = build_route("library.book", {"bid": bid}, {"tab": "chapters", "filter": qm.QA_FAILED_FILTER})
            _conn, session = tb._fake_session("android")
            screen = BookPageScreen(parse_route(route), tl._ctx(session.page, service))
            _mount(session.page, screen.get_body())
            screen.did_show()
            for _ in range(200):
                if screen.chapters.view is not None and screen.chapters.rows:
                    break
                await asyncio.sleep(0.02)
            session.page.update()
            chapters = screen.chapters
            assert screen.initial_tab == "chapters" and chapters.filter_group == "failed"
            visible = [row.status for row in chapters._filtered()]
            assert visible == ["qa_failed"]
            screen.dispose()

        asyncio.run(scenario())
    finally:
        lc.uninstall_library_env()


# ==========================================================================
# U6 second review round
# ==========================================================================


def _disabled(control) -> bool:
    """An ``action_button``: the button itself, or the button of its [button, ReasonChip] row."""
    import flet as ft

    return bool((control.controls[0] if isinstance(control, ft.Row) else control).disabled)


@needs_flet
def test_reopened_tool_screens_follow_the_queued_job_of_an_earlier_visit(tmp_path, monkeypatch):
    """A Headers / Converter / QA screen reopened while the job an earlier visit started is still queued or
    running keeps Stop and its disabled start buttons (no duplicate job); the Converter only follows a job
    of its own output folder."""
    monkeypatch.setitem(sys.modules, "qa_scan_runtime", fake_qa_runtime())
    from glossarion_mobile.services.jobs import JobSpec
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.converter import ConverterScreen
    from glossarion_mobile.ui.tools.headers_screen import HeadersScreen
    from glossarion_mobile.ui.tools.qa_screen import QaScannerScreen

    tb = _tb()
    target = tool_target(tmp_path, "Again")
    other = tool_target(tmp_path, "Elsewhere")

    async def scenario():
        _conn, session = tb._fake_session("android")
        page = session.page
        ctx = _ctx(page)
        jobs = ctx.jobs

        def open_screen(cls, route, context=ctx):
            screen = cls(parse_route(route), context)
            page.views[0].controls.clear()  # the previous visit's View left (the shell's pop is its own update)
            page.update()
            _mount(page, screen.get_body())
            screen.did_show()
            return screen

        # Headers & metadata: Translate Headers Now from the first visit is still queued
        first = open_screen(HeadersScreen, "/tools/headers")
        first.set_targets([target])
        header_job = await first.translate_headers()
        first.dispose()
        again = open_screen(HeadersScreen, "/tools/headers")
        await _settle()
        page.update()
        assert again.active_job is not None and again.active_job.id == header_job
        assert again.stop_button.visible and _disabled(again.header_actions.controls[0])
        again._on_stop()
        assert jobs.stops == [header_job]
        jobs.finish(header_job, state=JobState.CANCELLED)
        await _settle()
        assert not again.stop_button.visible and not _disabled(again.header_actions.controls[0])
        again.dispose()
        # Converter: a compile of the chosen folder (started on the first visit) is followed ...
        converter = open_screen(ConverterScreen, "/tools/convert")
        converter.set_target(target)
        compile_job = await converter.compile("epub")
        converter.dispose()
        reopened = open_screen(ConverterScreen, "/tools/convert")  # the tool state keeps the folder
        page.update()
        assert reopened.active_job.id == compile_job and reopened.stop_button.visible
        assert _disabled(reopened.actions_row.controls[0])  # Compile EPUB: "A compile job is running"
        reopened.dispose()
        # ... but not by a Converter on another folder, until that folder is chosen
        fresh = _ctx(page, jobs=jobs)
        elsewhere = open_screen(ConverterScreen, "/tools/convert", fresh)
        elsewhere.set_target(other)
        page.update()
        assert elsewhere.active_job is None and not elsewhere.stop_button.visible
        elsewhere.set_target(target)
        assert elsewhere.active_job.id == compile_job and elsewhere.stop_button.visible
        elsewhere.dispose()
        # QA Scanner: a queued scan (not only the running one) is followed too
        qa_job = await jobs.submit(JobSpec(kind="qa_scan", title="Again", inputs=(target.folder,),
                                           params={"targets": [{"folder": target.folder, "source": None}]}))
        qa = open_screen(QaScannerScreen, "/tools/qa")
        await _settle()
        page.update()
        assert qa.active_job.id == qa_job and qa.stop_button.visible and qa.start_button.disabled
        jobs.run(qa_job)
        await _settle()
        assert qa.stop_button.visible and qa.run_status.value.startswith("Running")
        qa.dispose()

    asyncio.run(scenario())
