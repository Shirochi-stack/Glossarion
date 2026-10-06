"""U7: the async batch core (async_batch_core) against the frozen async_api_processor.

Moved in milestone U7 out of ``async_api_processor.py`` (frozen here at ``U7_BASE_SHA`` through
``git show``) into the GUI-free ``async_batch_core``:

* ``AsyncAPIStatus`` / ``AsyncJobInfo`` / ``AsyncAPIProcessor`` byte for byte (``__init__`` gained
  ``jobs_file=None``, default unchanged);
* the ``AsyncProcessingDialog`` workflow methods as ``AsyncBatchJobMixin``: each body is the
  dialog's with the Qt statements replaced by hooks (``SUBSTITUTIONS`` below, the same table the
  mixin docstring lists); the dialog defines every hook with the original Qt statement;
* ``HeadlessAsyncBatch``: the same workflow without Qt (Glossarion Mobile).

Tiers:

V  verbatim AST: moved classes/functions equal the frozen source; every moved dialog method equals
   the frozen one after the hook substitutions; the kept view methods are unchanged (five use the
   shared row/model helpers instead of inline code); the dialog defines every hook with the Qt
   statement it replaces; every old module name still imports from async_api_processor.
R  request goldens per provider (OpenAI incl. o/GPT-5 token parameter, Anthropic, Gemini incl.
   thinking/safety env, Mistral, Groq) and the submit calls with mocked HTTP / a fake google-genai
   SDK: pinned values, and (with PySide6) equal to the frozen code's.
J  job file round trip (jobs_file parameter, reload, to_dict/from_dict) and byte equality with the
   frozen processor's file.
W  offscreen dialog workflow (estimate, submit, status, retrieve twice, Anthropic / Gemini /
   Mistral / Groq, cancel, delete, clear, unsupported model, AuthGPT without an sk- key): the
   frozen dialog and the rewired dialog give the same message boxes, logs, job files, provider
   calls and output workspaces; HeadlessAsyncBatch then reproduces the job files, provider calls
   and output workspaces of the desktop run.
H  headless semantics: questions via host.ask / preset answers, notices, polling until completion
   without stack growth, the stop latch, import without PySide6.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/test_async_batch_core.py
"""

from __future__ import annotations

import ast
import collections
import contextlib
import functools
import inspect
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
import types
import zipfile
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _p in (str(TESTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import async_batch_core as core  # noqa: E402
import requests  # noqa: E402
from _headless_env import headless_owner  # noqa: E402

#: The U6 commit: async_api_processor.py as it was before the U7 move.
U7_BASE_SHA = "41814faa95e273e956870bd3aac5a2c6fb7d66b1"

#: Dialog methods moved into AsyncBatchJobMixin (legacy class order).
MOVED = (
    "_get_opf_spine_map",
    "_check_selected_status", "_fetch_openai_error_snippet",
    "_retrieve_selected_results", "_cancel_selected_job",
    "_cancel_openai_job", "_cancel_anthropic_job", "_cancel_gemini_job", "_cancel_mistral_job", "_cancel_groq_job",
    "_estimate_cost", "_estimate_batch_cost", "count_tokens",
    "_start_processing", "_async_processing_worker", "_prepare_environment_variables",
    "_safe_int", "_extract_chapters_for_async",
    "_delete_selected_job", "_clear_completed_jobs",
    "_prepare_chapter_messages",
    "_submit_batch_sync", "_submit_gemini_batch_sync", "_submit_mistral_batch_sync", "_submit_groq_batch_sync",
    "_start_polling", "_handle_completed_job", "_show_error_details", "_extract_chapter_number",
    "_get_api_key_from_gui",
)
#: Kept view methods whose inline row / model / refresh code now calls the shared helpers.
EDITED_VIEW = {
    "_create_info_section": ("gui_model_name", "async_support_status"),
    "_refresh_model_info": ("gui_model_name", "async_support_status"),
    "_update_selected_job_progress": ("selected_job_progress",),
    "_refresh_jobs_list": ("job_display_row",),
    "_start_auto_refresh": ("refresh_pending_job_statuses",),
}

#: dialog code -> hook (regex, replacement), applied in order to the frozen method source.
SUBSTITUTIONS = (
    (r"QMessageBox\.Yes\b", "self._MB_YES"),
    (r"QMessageBox\.No\b", "self._MB_NO"),
    (r"QMessageBox\.Cancel\b", "self._MB_CANCEL"),
    (r"QMessageBox\.(warning|information|critical|question)\(\s*self\.dialog,\s*", r"self._async_msgbox('\1', "),
    (r"QTimer\.singleShot\(", "self._async_single_shot("),
    (r"QApplication\.processEvents\(\)", "self._async_process_events()"),
    (r"self\.cost_info_label\.setText\(", "self._async_set_cost_info("),
    (r"self\.start_button\.setEnabled\(", "self._async_set_start_enabled("),
    (r"self\.wait_for_completion_checkbox\.isChecked\(\)", "self._async_wait_for_completion()"),
    (r"self\.poll_interval_spinbox\.value\(\)", "self._async_poll_interval()"),
    (r"self\.dialog\.setCursor\(Qt\.WaitCursor\)", "self._async_set_wait_cursor(True)"),
    (r"self\.dialog\.setCursor\(Qt\.ArrowCursor\)", "self._async_set_wait_cursor(False)"),
    (r"hasattr\(self, 'dialog'\) and self\.dialog\.isVisible\(\)", "self._async_dialog_visible()"),
)

#: hook -> the Qt statement its desktop override must run (AST of the override's last statement).
DESKTOP_HOOKS = {
    "_async_msgbox": "return getattr(QMessageBox, kind)(self.dialog, *args)",
    "_async_single_shot": "return QTimer.singleShot(*args)",
    "_async_process_events": "QApplication.processEvents()",
    "_async_set_cost_info": "self.cost_info_label.setText(text)",
    "_async_set_start_enabled": "self.start_button.setEnabled(enabled)",
    "_async_wait_for_completion": "return self.wait_for_completion_checkbox.isChecked()",
    "_async_poll_interval": "return self.poll_interval_spinbox.value()",
    "_async_set_wait_cursor": "self.dialog.setCursor(Qt.WaitCursor if waiting else Qt.ArrowCursor)",
    "_async_dialog_visible": "return hasattr(self, 'dialog') and self.dialog.isVisible()",
}

SECRET = "sk-test-0123456789"


# =============================================================================================
# frozen source
# =============================================================================================

@functools.lru_cache(maxsize=None)
def _git_show(relpath):
    try:
        data = subprocess.run(["git", "show", f"{U7_BASE_SHA}:{relpath}"], cwd=str(REPO_ROOT),
                              capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, str(exc)
    return data.decode("utf-8-sig").replace("\r\n", "\n"), None


def legacy_text():
    text, error = _git_show("src/async_api_processor.py")
    if text is None:
        pytest.skip(f"async_api_processor.py@{U7_BASE_SHA[:8]} unavailable: {error}")
    return text


def live_text(name):
    return (SRC / name).read_bytes().decode("utf-8-sig").replace("\r\n", "\n")


def load_legacy_module(file_path):
    """Execute the frozen async_api_processor (needs PySide6) with ``__file__ = file_path``."""
    pytest.importorskip("PySide6.QtWidgets")
    module = types.ModuleType("async_api_processor")
    module.__file__ = str(file_path)
    exec(compile(legacy_text(), str(SRC / "async_api_processor.py"), "exec"), module.__dict__)
    return module


def substitute(src):
    for pattern, replacement in SUBSTITUTIONS:
        src = re.sub(pattern, replacement, src)
    return src


def _class(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def _methods(text, class_name):
    cls = _class(ast.parse(text), class_name)
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def _segment(text, node):
    lines = text.split("\n")
    start = node.lineno - 1 - len(getattr(node, "decorator_list", []) or [])
    return textwrap.dedent("\n".join(lines[start:node.end_lineno]))


def _dump(src):
    return ast.dump(ast.parse(textwrap.dedent(src)))


# =============================================================================================
# tier V: verbatim moves
# =============================================================================================

def test_core_classes_are_the_frozen_ones():
    old, new = ast.parse(legacy_text()), ast.parse(live_text("async_batch_core.py"))
    for name in ("AsyncAPIStatus", "AsyncJobInfo"):
        assert ast.dump(_class(old, name)) == ast.dump(_class(new, name)), name
    old_proc, new_proc = _class(old, "AsyncAPIProcessor"), _class(new, "AsyncAPIProcessor")
    old_body = [n for n in old_proc.body if not (isinstance(n, ast.FunctionDef) and n.name == "__init__")]
    new_body = [n for n in new_proc.body if not (isinstance(n, ast.FunctionDef) and n.name == "__init__")]
    assert [ast.dump(n) for n in old_body] == [ast.dump(n) for n in new_body]


def test_processor_init_only_adds_the_jobs_file_parameter():
    old = _methods(legacy_text(), "AsyncAPIProcessor")["__init__"]
    new = _methods(live_text("async_batch_core.py"), "AsyncAPIProcessor")["__init__"]
    assert [a.arg for a in new.args.args] == ["self", "gui_instance", "jobs_file"]
    assert [ast.dump(d) for d in new.args.defaults] == [ast.dump(ast.Constant(None))]

    class _Unwrap(ast.NodeTransformer):
        def visit_BoolOp(self, node):  # ``jobs_file or <legacy default>`` -> <legacy default>
            if (isinstance(node.op, ast.Or) and len(node.values) == 2 and isinstance(node.values[0], ast.Name)
                    and node.values[0].id == "jobs_file"):
                return node.values[1]
            return node

    new_stmts = [ast.dump(_Unwrap().visit(s)) for s in new.body[1:]]
    assert new_stmts == [ast.dump(s) for s in old.body[1:]]


def test_module_helpers_and_optional_imports_are_the_frozen_ones():
    old, new = ast.parse(legacy_text()), ast.parse(live_text("async_batch_core.py"))
    for name in ("_is_antigravity_model_name", "_clamp_output_tokens_for_selected_model"):
        pick = lambda tree: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)  # noqa: E731
        assert ast.dump(pick(old)) == ast.dump(pick(new)), name
    tries = lambda tree: [ast.dump(n) for n in tree.body if isinstance(n, ast.Try)]  # noqa: E731
    old_tries = [t for t in tries(old) if "PySide6" not in t]
    assert old_tries == tries(new)


@pytest.mark.parametrize("name", MOVED)
def test_moved_dialog_method_is_verbatim_up_to_the_hook_table(name):
    text = legacy_text()
    old = _methods(text, "AsyncProcessingDialog")[name]
    new_text = live_text("async_batch_core.py")
    new = _methods(new_text, "AsyncBatchJobMixin")[name]
    assert _dump(substitute(_segment(text, old))) == _dump(_segment(new_text, new))
    leftover = re.findall(r"QMessageBox|QTimer|QApplication|\bQt\.|self\.dialog\.|cost_info_label|start_button\."
                          r"|poll_interval_spinbox|wait_for_completion_checkbox", _segment(new_text, new))
    assert not leftover, leftover


def test_kept_dialog_methods_are_unchanged():
    text = legacy_text()
    old = _methods(text, "AsyncProcessingDialog")
    new_text = live_text("async_api_processor.py")
    new = _methods(new_text, "AsyncProcessingDialog")
    kept = [n for n in old if n not in MOVED]
    assert set(kept) <= set(new), set(kept) - set(new)
    for name in kept:
        if name in EDITED_VIEW:
            for helper in EDITED_VIEW[name]:
                assert helper in _segment(new_text, new[name]), (name, helper)
            continue
        assert _dump(_segment(text, old[name])) == _dump(_segment(new_text, new[name])), name
    assert set(new) - set(kept) == set(DESKTOP_HOOKS)
    for name in ("_show_context_menu", "_get_selected_job_ids", "_on_job_select", "_log", "_show_error",
                 "_show_info", "_show_warning"):
        assert name in kept


def test_module_functions_of_the_dialog_module_are_unchanged():
    text = legacy_text()
    old = {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    new_text = live_text("async_api_processor.py")
    new = {n.name: n for n in ast.parse(new_text).body if isinstance(n, ast.FunctionDef)}
    for name in ("_prewarm_dialog_offscreen", "show_async_processing_dialog", "add_async_processing_button"):
        assert _dump(_segment(text, old[name])) == _dump(_segment(new_text, new[name])), name


def test_desktop_dialog_defines_every_hook_with_its_qt_statement():
    new_text = live_text("async_api_processor.py")
    methods = _methods(new_text, "AsyncProcessingDialog")
    for hook, statement in DESKTOP_HOOKS.items():
        body = [s for s in methods[hook].body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        assert len(body) == 1 and ast.dump(body[0]) == ast.dump(ast.parse(statement).body[0]), hook
    # every hook / button constant the moved bodies use is overridden by the dialog class body
    mixin_text = live_text("async_batch_core.py")
    used = set()
    for name in MOVED:
        used |= set(re.findall(r"self\.(_async_\w+|_MB_\w+)", _segment(mixin_text, _methods(mixin_text, "AsyncBatchJobMixin")[name])))
    import async_api_processor

    used -= set(MOVED)
    for name in used:
        assert name in vars(async_api_processor.AsyncProcessingDialog), name
    for name in ("_MB_OK", "_MB_YES", "_MB_NO", "_MB_CANCEL"):
        assert isinstance(vars(async_api_processor.AsyncProcessingDialog)[name], property), name


def test_every_old_module_name_still_imports():
    import async_api_processor

    tree = ast.parse(legacy_text())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.Try):
            statements = node.body + node.orelse + node.finalbody + [s for h in node.handlers for s in h.body]
            for sub in statements:
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    names |= {(a.asname or a.name).split(".")[0] for a in sub.names}
                elif isinstance(sub, ast.FunctionDef):
                    names.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    names |= {t.id for t in sub.targets if isinstance(t, ast.Name)}
    optional = {"genai": core.HAS_GEMINI, "anthropic": core.HAS_ANTHROPIC, "openai": core.HAS_OPENAI}
    for name in sorted(names):
        if name in optional and not optional[name]:
            continue
        assert hasattr(async_api_processor, name), name
    for name in ("AsyncAPIStatus", "AsyncJobInfo", "AsyncAPIProcessor", "_clamp_output_tokens_for_selected_model",
                 "_clamp_antigravity_output_tokens", "_is_antigravity_model_name", "TextFileProcessor", "tiktoken",
                 "HAS_GEMINI", "HAS_ANTHROPIC", "HAS_OPENAI"):
        assert getattr(async_api_processor, name) is getattr(core, name), name
    assert issubclass(async_api_processor.AsyncProcessingDialog, core.AsyncBatchJobMixin)


def test_core_is_python310_and_line_endings_are_uniform():
    for name in ("async_batch_core.py", "async_api_processor.py"):
        data = (SRC / name).read_bytes()
        ast.parse(data.decode("utf-8-sig"), feature_version=(3, 10))
        assert data.count(b"\r\n") in (0, data.count(b"\n")), name


def test_core_and_dialog_module_import_without_pyside6():
    probe = textwrap.dedent(f"""
        import sys
        for name in ("PySide6", "shiboken6", "tkinter", "translator_gui", "dpi_setup"):
            sys.modules[name] = None
        sys.path.insert(0, {str(SRC)!r})
        import async_batch_core, async_api_processor
        assert async_api_processor.QMessageBox is None
        assert async_api_processor.AsyncAPIProcessor is async_batch_core.AsyncAPIProcessor
        assert async_batch_core.HeadlessAsyncBatch
        loaded = [n for n, m in sys.modules.items() if m is not None and n.split(".")[0] in ("PySide6", "translator_gui", "dpi_setup")]
        assert not loaded, loaded
        print("OK")
    """)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, encoding="utf-8",
                            env=env, cwd=str(SRC), timeout=240)
    assert result.returncode == 0 and "OK" in result.stdout, result.stdout[-2000:] + result.stderr[-4000:]


# =============================================================================================
# fakes: HTTP providers, google-genai SDK, EPUB fixture
# =============================================================================================

class _Resp:
    def __init__(self, status, payload=None, text=None):
        self.status_code = status
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        return self._payload


class FakeProviders:
    """Batch endpoints of OpenAI / Anthropic / Mistral / Groq-on-OpenAI with deterministic ids."""

    def __init__(self):
        self.calls = []
        self.counter = 0
        self.uploads = {}  # openai input file id -> request dicts
        self.batches = {}  # openai batch id -> input file id
        self.anthropic = {}  # anthropic batch id -> request dicts
        self.openai_state = "completed"

    def _next(self):
        self.counter += 1
        return self.counter

    def post(self, url, headers=None, files=None, data=None, json=None, timeout=None, **kw):  # noqa: A002
        entry = {"method": "POST", "url": url, "headers": dict(headers or {})}
        if files:
            name, handle, mime = files["file"]
            lines = [__import__("json").loads(line) for line in handle.read().decode("utf-8").splitlines() if line.strip()]
            entry.update(file=[name, mime, lines], data=dict(data or {}))
            self.calls.append(entry)
            file_id = f"file-in-{self._next()}"
            self.uploads[file_id] = lines
            return _Resp(200, {"id": file_id, "purpose": "batch"})
        entry["json"] = json
        self.calls.append(entry)
        if url == "https://api.openai.com/v1/batches":
            batch_id = f"batch_{self._next()}"
            self.batches[batch_id] = json["input_file_id"]
            return _Resp(200, {"id": batch_id, "status": "validating", "input_file_id": json["input_file_id"]})
        if url.startswith("https://api.openai.com/v1/batches/") and url.endswith("/cancel"):
            return _Resp(200, {"status": "cancelling"})
        if url == "https://api.anthropic.com/v1/messages/batches":
            batch_id = f"msgbatch_{self._next()}"
            self.anthropic[batch_id] = json["requests"]
            return _Resp(200, {"id": batch_id, "processing_status": "in_progress"})
        if url == "https://api.mistral.ai/v1/batch/jobs":
            return _Resp(200, {"id": f"mistral_{self._next()}", "status": "QUEUED"})
        if url.endswith(":cancel"):
            return _Resp(200, {})
        return _Resp(404, {"error": "unknown"}, "unknown endpoint")

    def get(self, url, headers=None, timeout=None, **kw):
        self.calls.append({"method": "GET", "url": url, "headers": dict(headers or {})})
        prefix = "https://api.openai.com/v1/batches/"
        if url.startswith(prefix) and url[len(prefix):] in self.batches:
            batch_id = url[len(prefix):]
            total = len(self.uploads[self.batches[batch_id]])
            return _Resp(200, {"id": batch_id, "status": self.openai_state,
                               "request_counts": {"total": total, "completed": total, "failed": 0},
                               "output_file_id": f"file-out-{batch_id}"})
        match = re.match(r"https://api\.openai\.com/v1/files/file-out-(batch_\d+)/content$", url)
        if match:
            lines = []
            for i, req in enumerate(self.uploads[self.batches[match.group(1)]]):
                content = (f"Plain line {i}\nSecond line" if i == 1 else
                           f"<html><body><h1>EN</h1><p>{req['custom_id']}</p></body></html>")
                lines.append(json.dumps({"custom_id": req["custom_id"], "response": {"body": {"choices": [
                    {"message": {"content": content}, "finish_reason": "stop"}]}}}))
            return _Resp(200, None, "\n".join(lines))
        match = re.match(r"https://api\.anthropic\.com/v1/messages/batches/(msgbatch_\d+)(/results)?$", url)
        if match and match.group(1) in self.anthropic:
            reqs = self.anthropic[match.group(1)]
            if not match.group(2):
                return _Resp(200, {"id": match.group(1), "processing_status": "ended",
                                   "results_summary": {"succeeded": len(reqs), "failed": 0, "total": len(reqs)},
                                   "results_url": url + "/results"})
            lines = [json.dumps({"custom_id": r["custom_id"], "result": {"type": "succeeded", "message": {
                "content": [{"text": f"<p>Claude {r['custom_id']}</p>"}], "stop_reason": "end_turn"}}}) for r in reqs]
            return _Resp(200, None, "\n".join(lines))
        return _Resp(404, {"error": "unknown"}, "unknown endpoint")

    def delete(self, url, headers=None, **kw):
        self.calls.append({"method": "DELETE", "url": url, "headers": dict(headers or {})})
        return _Resp(200, {})


def install_fake_genai(monkeypatch, recorder):
    """A google-genai stand-in recording uploads, batch creation, status and downloads."""
    genai = types.ModuleType("google.genai")
    gtypes = types.ModuleType("google.genai.types")
    state = {"uploads": {}, "batches": {}}

    class UploadFileConfig:
        def __init__(self, mime_type=None, display_name=None):
            self.mime_type, self.display_name = mime_type, display_name

    class _Files:
        def upload(self, file=None, config=None):
            with open(file, "r", encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh.read().splitlines() if line.strip()]
            name = f"files/up-{len(state['uploads']) + 1}"
            state["uploads"][name] = lines
            recorder.append(["upload", lines, config.mime_type, config.display_name])
            return types.SimpleNamespace(name=name)

        def download(self, file=None):
            recorder.append(["download", file])
            batch = next(b for b in state["batches"].values() if b["out"] == file)
            out = [json.dumps({"key": line["key"], "response": {"candidates": [
                {"content": {"parts": [{"text": f"<p>Gemini {line['key']}</p>"}]}}]}})
                for line in state["uploads"][batch["src"]]]
            return "\n".join(out).encode("utf-8")

    class _Batches:
        def create(self, model=None, src=None, config=None):
            name = f"batches/gem-{len(state['batches']) + 1}"
            state["batches"][name] = {"src": src, "out": f"files/out-{len(state['batches']) + 1}"}
            recorder.append(["create", model, src, config])
            return types.SimpleNamespace(name=name, state=types.SimpleNamespace(name="JOB_STATE_PENDING"))

        def get(self, name=None):
            recorder.append(["get", name])
            return types.SimpleNamespace(name=name, state=types.SimpleNamespace(name="JOB_STATE_SUCCEEDED"),
                                         dest=types.SimpleNamespace(file_name=state["batches"][name]["out"]))

    class Client:
        def __init__(self, api_key=None):
            recorder.append(["client", api_key])
            self.files, self.batches = _Files(), _Batches()

    gtypes.UploadFileConfig = UploadFileConfig
    genai.Client = Client
    genai.types = gtypes
    try:
        import google as google_pkg
    except Exception:
        google_pkg = types.ModuleType("google")
        google_pkg.__path__ = []
        monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)
    monkeypatch.setattr(google_pkg, "genai", genai, raising=False)
    return state


CHAPTER_WORDS = (("c2.xhtml", "Chapter 2", "둘"), ("c1.xhtml", "Chapter 1", "하나"), ("c3.xhtml", "Chapter 3", "셋"))


def build_epub(path):
    """A 3-chapter EPUB whose spine order (c2, c1, c3) differs from the name order."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", (
            '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
            '</rootfiles></container>'))
        manifest = "".join(f'<item id="{n[:-6]}" href="{n}" media-type="application/xhtml+xml"/>' for n, _, _ in CHAPTER_WORDS)
        spine = "".join(f'<itemref idref="{n[:-6]}"/>' for n, _, _ in CHAPTER_WORDS)
        zf.writestr("OEBPS/content.opf", (
            '<?xml version="1.0" encoding="utf-8"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0" '
            'unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Test Book</dc:title>'
            '<dc:creator>Author</dc:creator><dc:language>ko</dc:language><dc:identifier id="id">x1</dc:identifier>'
            '</metadata><manifest><item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
            '<item id="css" href="style.css" media-type="text/css"/><item id="img" href="images/pic.png" '
            f'media-type="image/png"/>{manifest}</manifest><spine toc="ncx">{spine}</spine></package>'))
        zf.writestr("OEBPS/toc.ncx", ('<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" '
                                       'version="2005-1"><head></head><docTitle><text>T</text></docTitle><navMap></navMap></ncx>'))
        zf.writestr("OEBPS/style.css", "p { margin: 0 }")
        zf.writestr("OEBPS/images/pic.png", b"\x89PNG\r\n\x1a\nfake")
        for name, title, word in CHAPTER_WORDS:
            body = "".join(f"<p>{word} 문장 {i} 입니다.</p>" for i in range(60))
            zf.writestr(f"OEBPS/{name}", (
                f'<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml"><head>'
                f'<title>{title}</title></head><body><h1>{title}</h1>{body}</body></html>'))


_TS = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?")
#: traceback frames name the module the code lives in (async_api_processor before, async_batch_core now)
_FRAME = re.compile(r'File "[^"]+", line \d+, in ')
_STAMP = re.compile(r"\d{8}_\d{6}")


def norm(value, roots=()):
    """Timestamps -> <ts>, the side's sandbox roots -> <root>, recursively."""
    if isinstance(value, dict):
        return {norm(k, roots): norm(v, roots) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [norm(v, roots) for v in value]
    if isinstance(value, str):
        for root in roots:
            for form in {str(root), str(root).replace("\\", "/"), str(root).replace("\\", "\\\\")}:
                value = value.replace(form, "<root>")
        value = _FRAME.sub("File <src>, in ", _STAMP.sub("<stamp>", _TS.sub("<ts>", value)))
        if "Traceback (most recent call last)" in value:  # frames only: source lines need linecache
            value = "\n".join(line for line in value.split("\n") if not line.startswith("    "))
        return value
    return value


def tree_snapshot(folder, roots=()):
    """{relative path: content} of an output workspace (JSON parsed and normalised)."""
    out = {}
    folder = Path(folder)
    if not folder.is_dir():
        return out
    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(folder).as_posix()
        if path.is_dir():
            out[rel + "/"] = None
        elif path.suffix == ".json":
            out[rel] = norm(json.loads(path.read_text(encoding="utf-8")), roots)
        else:
            out[rel] = path.read_bytes().replace(b"\r\n", b"\n")
    return out


def jobs_snapshot(path, roots=()):
    path = Path(path)
    return norm(json.loads(path.read_text(encoding="utf-8")), roots) if path.exists() else None


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Sandbox: Library / output roots, HOME and the async env all point into tmp_path."""
    for key in ("ENABLE_GEMINI_THINKING", "GEMINI_THINKING_LEVEL", "THINKING_BUDGET", "DISABLE_GEMINI_SAFETY",
                "OUTPUT_DIRECTORY", "GLOSSARY_SHARED_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "library"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    providers = FakeProviders()
    monkeypatch.setattr(requests, "post", providers.post)
    monkeypatch.setattr(requests, "get", providers.get)
    monkeypatch.setattr(requests, "delete", providers.delete)
    return providers


# =============================================================================================
# tier R: request goldens per provider
# =============================================================================================

CHAPTERS = [
    {"id": "0001_c1", "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "<p>하나</p>"}],
     "temperature": 0.3, "max_tokens": 4096},
    {"id": "0002_c2", "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "  <p>둘</p>  "},
                                   {"role": "assistant", "content": ""}],
     "temperature": 0.5, "max_tokens": 8192},
]

OPENAI_BODY = lambda model, token_param, cid, user, temp, limit: {  # noqa: E731
    "custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
    "body": {"model": model, "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": user}],
             "temperature": temp, token_param: limit}}

GEMINI_REQUEST = lambda cid, prompt, temp, limit, extra=None: {  # noqa: E731
    "custom_id": cid, "generateContentRequest": dict({
        "model": "models/gemini-2.5-flash", "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": dict({"temperature": temp, "maxOutputTokens": limit}, **(extra or {}))})}
SAFETY = [{"category": c, "threshold": "BLOCK_NONE"} for c in (
    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT", "HARM_CATEGORY_CIVIC_INTEGRITY")]

#: (model, env) -> expected prepare_batch_request result (pinned desktop behaviour, bugs included:
#: gpt-4o-mini is sent as gpt-4o - DISCREPANCIES "U7 async batch").
REQUEST_GOLDENS = {
    ("gpt-4o-mini", ()): {"requests": [
        OPENAI_BODY("gpt-4o", "max_tokens", "0001_c1", "<p>하나</p>", 0.3, 4096),
        OPENAI_BODY("gpt-4o", "max_tokens", "0002_c2", "<p>둘</p>", 0.5, 8192)]},
    ("gpt-5-mini", ()): {"requests": [
        OPENAI_BODY("gpt-5-mini", "max_completion_tokens", "0001_c1", "<p>하나</p>", 0.3, 4096),
        OPENAI_BODY("gpt-5-mini", "max_completion_tokens", "0002_c2", "<p>둘</p>", 0.5, 8192)]},
    ("groq/llama-3.1-8b-instant", ()): {"requests": [
        OPENAI_BODY("groq/llama-3.1-8b-instant", "max_tokens", "0001_c1", "<p>하나</p>", 0.3, 4096),
        OPENAI_BODY("groq/llama-3.1-8b-instant", "max_tokens", "0002_c2", "<p>둘</p>", 0.5, 8192)]},
    ("claude-sonnet-4-5", ()): {"requests": [
        {"custom_id": "0001_c1", "params": {"model": "claude-sonnet-4-5", "messages": [
            {"role": "user", "content": "<p>하나</p>"}], "max_tokens": 4096, "temperature": 0.3, "system": "SYS"}},
        {"custom_id": "0002_c2", "params": {"model": "claude-sonnet-4-5", "messages": [
            {"role": "user", "content": "  <p>둘</p>  "}, {"role": "assistant", "content": ""}],
            "max_tokens": 8192, "temperature": 0.5, "system": "SYS"}}]},
    ("mistral-large-latest", ()): {"requests": [
        {"custom_id": "0001_c1", "model": "mistral-large-latest", "messages": CHAPTERS[0]["messages"],
         "temperature": 0.3, "max_tokens": 4096},
        {"custom_id": "0002_c2", "model": "mistral-large-latest", "messages": CHAPTERS[1]["messages"],
         "temperature": 0.5, "max_tokens": 8192}]},
    ("gemini-2.5-flash", ()): {"requests": [
        GEMINI_REQUEST("0001_c1", "INSTRUCTIONS: SYS\n\nUSER: <p>하나</p>", 0.3, 4096),
        GEMINI_REQUEST("0002_c2", "INSTRUCTIONS: SYS\n\nUSER:   <p>둘</p>  \n\nASSISTANT: ", 0.5, 8192)]},
    ("gemini-2.5-flash", (("THINKING_BUDGET", "2048"), ("GEMINI_THINKING_LEVEL", "high"),
                          ("DISABLE_GEMINI_SAFETY", "true"))): {"requests": [
        dict(GEMINI_REQUEST("0001_c1", "INSTRUCTIONS: SYS\n\nUSER: <p>하나</p>", 0.3, 4096,
                            {"thinking": {"level": "high", "budgetTokens": 2048}})),
        dict(GEMINI_REQUEST("0002_c2", "INSTRUCTIONS: SYS\n\nUSER:   <p>둘</p>  \n\nASSISTANT: ", 0.5, 8192,
                            {"thinking": {"level": "high", "budgetTokens": 2048}}))]},
    ("gemini-2.5-flash", (("ENABLE_GEMINI_THINKING", "0"), ("THINKING_BUDGET", "2048"))): {"requests": [
        GEMINI_REQUEST("0001_c1", "INSTRUCTIONS: SYS\n\nUSER: <p>하나</p>", 0.3, 4096),
        GEMINI_REQUEST("0002_c2", "INSTRUCTIONS: SYS\n\nUSER:   <p>둘</p>  \n\nASSISTANT: ", 0.5, 8192)]},
}
for _req in REQUEST_GOLDENS[("gemini-2.5-flash", (("THINKING_BUDGET", "2048"), ("GEMINI_THINKING_LEVEL", "high"),
                                                  ("DISABLE_GEMINI_SAFETY", "true")))]["requests"]:
    _req["generateContentRequest"]["safetySettings"] = SAFETY


def _prepare(processor_cls, model, monkeypatch, env):
    for key, value in env:
        monkeypatch.setenv(key, value)
    processor = object.__new__(processor_cls)
    processor.gui = None
    return processor.prepare_batch_request(json.loads(json.dumps(CHAPTERS)), model)


@pytest.mark.parametrize("model,env", list(REQUEST_GOLDENS), ids=lambda v: str(v) if isinstance(v, str) else "-".join(k for k, _ in v) or "default")
def test_request_building_golden_per_provider(model, env, isolated, monkeypatch, tmp_path, capsys):
    got = _prepare(core.AsyncAPIProcessor, model, monkeypatch, env)
    assert got == REQUEST_GOLDENS[(model, env)]
    if importlib_util_find("PySide6"):
        legacy = load_legacy_module(tmp_path / "legacy" / "async_api_processor.py")
        assert _prepare(legacy.AsyncAPIProcessor, model, monkeypatch, env) == got


def importlib_util_find(name):
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _submitter(module, gui):
    """A dialog-shaped object of ``module`` (frozen dialog or HeadlessAsyncBatch) for the submit calls."""
    if module is core:
        obj = object.__new__(core.HeadlessAsyncBatch)
    else:
        obj = object.__new__(module.AsyncProcessingDialog)
    obj.gui = gui
    obj.processor = module.AsyncAPIProcessor(gui, jobs_file=None) if module is core else object.__new__(module.AsyncAPIProcessor)
    obj.processor.gui = gui
    obj.processor.jobs = {}
    return obj


SUBMIT_PROVIDERS = {
    "gpt-4o-mini": ("openai", "https://api.openai.com/v1/files"),
    "groq/llama-3.1-8b-instant": ("openai", "https://api.openai.com/v1/files"),
    "claude-sonnet-4-5": ("anthropic", "https://api.anthropic.com/v1/messages/batches"),
    "mistral-large-latest": ("mistral", "https://api.mistral.ai/v1/batch/jobs"),
    "gemini-2.5-flash": ("gemini", None),
}


def _submit_once(module, model, monkeypatch, tmp_path):
    providers = FakeProviders()
    monkeypatch.setattr(requests, "post", providers.post)
    monkeypatch.setattr(requests, "get", providers.get)
    sdk = []
    install_fake_genai(monkeypatch, sdk)
    gui = types.SimpleNamespace(file_path=str(tmp_path / "Book.epub"), config={})
    if module is core:
        monkeypatch.setattr(core, "__file__", str(tmp_path / "core" / "async_batch_core.py"))
    obj = _submitter(module, gui)
    batch = obj.processor.prepare_batch_request(json.loads(json.dumps(CHAPTERS)), model)
    job = obj._submit_batch_sync(batch, model, SECRET)
    return {"job": norm(job.to_dict()), "calls": providers.calls, "sdk": norm(sdk)}


@pytest.mark.parametrize("model", list(SUBMIT_PROVIDERS))
def test_submission_golden_per_provider(model, isolated, monkeypatch, tmp_path, capsys):
    got = _submit_once(core, model, monkeypatch, tmp_path)
    provider, url = SUBMIT_PROVIDERS[model]
    job = got["job"]
    assert (job["provider"], job["model"], job["status"], job["total_requests"]) == (provider, model, "pending", 2)
    if provider == "openai":
        upload, create = got["calls"]
        assert upload["url"] == url and upload["headers"] == {"Authorization": f"Bearer {SECRET}"}
        assert upload["data"] == {"purpose": "batch"} and upload["file"][:2] == ["batch.jsonl", "application/jsonl"]
        assert upload["file"][2] == _prepare(core.AsyncAPIProcessor, model, monkeypatch, ())["requests"]
        assert create["url"] == "https://api.openai.com/v1/batches"
        assert create["headers"] == {"Authorization": f"Bearer {SECRET}", "Content-Type": "application/json"}
        assert create["json"] == {"input_file_id": "file-in-1", "endpoint": "/v1/chat/completions", "completion_window": "24h"}
        assert job["job_id"] == "batch_2" and job["metadata"]["file_id"] == "file-in-1"
        assert job["cost_estimate"] == core.AsyncAPIProcessor.estimate_cost(object.__new__(core.AsyncAPIProcessor), 2, 15000, model)[0]
    elif provider == "anthropic":
        (call,) = got["calls"]
        assert call["url"] == url and call["headers"] == {
            "X-API-Key": SECRET, "Content-Type": "application/json", "anthropic-version": "2023-06-01",
            "anthropic-beta": "message-batches-2024-09-24"}
        assert call["json"] == REQUEST_GOLDENS[(model, ())] and job["job_id"] == "msgbatch_1"
    elif provider == "mistral":
        (call,) = got["calls"]
        assert call["url"] == url and call["headers"] == {"Authorization": f"Bearer {SECRET}", "Content-Type": "application/json"}
        assert call["json"] == REQUEST_GOLDENS[(model, ())] and job["job_id"] == "mistral_1"
    else:
        assert got["calls"] == []
        client, upload, create = got["sdk"]
        assert client == ["client", SECRET]
        lines = upload[1]
        assert [line["key"] for line in lines] == ["0001_c1", "0002_c2"]
        assert lines[0]["request"] == {"contents": [{"parts": [{"text": "INSTRUCTIONS: SYS\n\nUSER: <p>하나</p>"}]}],
                                       "generation_config": {"temperature": 0.3, "maxOutputTokens": 4096}}
        assert upload[2:] == ["application/jsonl", "batch_requests_<stamp>.jsonl"]
        assert create == ["create", model, "files/up-1", {"display_name": "glossarion_batch_<stamp>"}]
        assert job["job_id"] == "batches/gem-1" and job["cost_estimate"] == 0.0
        assert job["metadata"] == {"batch_info": {"name": "batches/gem-1", "state": "JOB_STATE_PENDING",
                                                  "src_file": "files/up-1"},
                                   "source_file": str(tmp_path / "Book.epub")}
    if importlib_util_find("PySide6"):
        legacy = load_legacy_module(tmp_path / "legacy" / "async_api_processor.py")
        assert _submit_once(legacy, model, monkeypatch, tmp_path) == got


def test_gemini_thinking_is_stripped_from_the_batch_file(isolated, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("THINKING_BUDGET", "512")
    monkeypatch.setenv("DISABLE_GEMINI_SAFETY", "true")
    got = _submit_once(core, "gemini-2.5-flash", monkeypatch, tmp_path)
    lines = got["sdk"][1][1]
    assert all("thinking" not in line["request"]["generation_config"] for line in lines)
    assert all(line["request"]["safety_settings"] == SAFETY for line in lines)


# =============================================================================================
# tier J: job file
# =============================================================================================

def _sample_jobs(module):
    from datetime import datetime

    jobs = {}
    for i, status in enumerate(module.AsyncAPIStatus):
        jobs[f"job_{i}"] = module.AsyncJobInfo(
            job_id=f"job_{i}", provider=("openai", "anthropic", "gemini")[i % 3], model="gpt-4o", status=status,
            created_at=datetime(2026, 1, 2, 3, 4, 5), updated_at=datetime(2026, 1, 2, 3, 5, 6, 789),
            total_requests=10 + i, completed_requests=i, failed_requests=i % 2, cost_estimate=0.25 * i,
            input_file=None, output_file=f"out-{i}" if i % 2 else None, error_message=None,
            metadata={"source_file": f"C:/books/Book {i}.epub", "chapter_mapping": {"0001_c1": {"chapter_num": 1}},
                      "env": {"MODEL": "gpt-4o", "TEXT_EXTRACTION_METHOD": "standard"}, "한글": "값"})
    return jobs


def test_job_file_round_trip_and_default_path(isolated, tmp_path):
    jobs_file = tmp_path / "data" / "async_jobs.json"
    jobs_file.parent.mkdir()
    processor = core.AsyncAPIProcessor(None, jobs_file=str(jobs_file))
    assert processor.jobs == {}
    processor.jobs = _sample_jobs(core)
    processor._save_jobs()
    again = core.AsyncAPIProcessor(None, jobs_file=str(jobs_file))
    assert {k: v.to_dict() for k, v in again.jobs.items()} == {k: v.to_dict() for k, v in processor.jobs.items()}
    assert [type(v.status) for v in again.jobs.values()] == [core.AsyncAPIStatus] * len(core.AsyncAPIStatus)
    # the default stays next to the module (the frozen dialog path), mobile passes its own
    assert core.AsyncAPIProcessor.__init__.__defaults__ == (None,)
    src = inspect.getsource(core.AsyncAPIProcessor.__init__)
    assert "jobs_file or os.path.join(os.path.dirname(__file__), 'async_jobs.json')" in src
    # a broken entry is skipped, the rest loads (frozen behaviour)
    data = json.loads(jobs_file.read_text(encoding="utf-8"))
    data["job_0"]["status"] = "bogus"
    jobs_file.write_text(json.dumps(data), encoding="utf-8")
    assert "job_0" not in core.AsyncAPIProcessor(None, jobs_file=str(jobs_file)).jobs


def test_job_file_bytes_equal_the_frozen_processor(isolated, tmp_path):
    legacy = load_legacy_module(tmp_path / "legacy" / "async_api_processor.py")
    old = object.__new__(legacy.AsyncAPIProcessor)
    old.jobs_file = str(tmp_path / "old.json")
    old.jobs = _sample_jobs(legacy)
    old._save_jobs()
    new = core.AsyncAPIProcessor(None, jobs_file=str(tmp_path / "new.json"))
    new.jobs = _sample_jobs(core)
    new._save_jobs()
    assert (tmp_path / "old.json").read_bytes() == (tmp_path / "new.json").read_bytes()
    # and the frozen default path is the module folder, exactly like the core's
    frozen_default = legacy.AsyncAPIProcessor(None)
    assert frozen_default.jobs_file == str(tmp_path / "legacy" / "async_jobs.json").replace("/", os.sep)


def test_default_jobs_file_uses_the_mobile_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOSSARION_DATA_DIR", str(tmp_path / "appdata"))
    assert core.default_jobs_file() == os.path.join(str(tmp_path / "appdata"), "async_jobs.json")
    monkeypatch.delenv("GLOSSARION_DATA_DIR")
    assert core.default_jobs_file() == os.path.join(os.path.dirname(os.path.abspath(core.__file__)), "async_jobs.json")


# =============================================================================================
# tier W: offscreen dialog workflow, frozen vs rewired vs headless
# =============================================================================================

#: dialog question title -> answer (the BoxRecorder and HeadlessAsyncBatch.answers use the same script)
ANSWERS = {"Start Async Processing": "yes", "Possibly Unsupported": "no", "Gemini Batch API": "yes",
           "Cancel Jobs": "yes", "Confirm Delete": "yes", "Clear Completed Jobs": "yes"}


class BoxRecorder:
    def __init__(self, QMessageBox):
        self.Q = QMessageBox
        self.events = []
        self.answers = dict(ANSWERS)

    def install(self, monkeypatch):
        for kind in ("warning", "information", "critical", "question"):
            monkeypatch.setattr(self.Q, kind, staticmethod(self._box(kind)))

    def _box(self, kind):
        def box(parent, title, text, buttons=None, *rest):
            answer = None
            if buttons is not None:
                answer = self.answers.get(title, "yes")
            self.events.append([kind, title, text, answer])
            if answer is None:
                return self.Q.Ok
            return {"yes": self.Q.Yes, "no": self.Q.No, "cancel": self.Q.Cancel}[answer]
        return box


class Side:
    """One front end (frozen dialog, rewired dialog or HeadlessAsyncBatch) over a sandbox root."""

    def __init__(self, name, root, owner, kind, module, app=None, providers=None):
        self.name, self.root, self.owner, self.kind, self.module, self.app = name, Path(root), owner, kind, module, app
        self.providers = providers
        self.out_root = self.root / "out"
        if kind == "headless":
            self.obj = core.HeadlessAsyncBatch(owner, host=owner.host, jobs_file=str(self.root / "async_jobs.json"),
                                               answers=dict(ANSWERS))
        else:
            self.obj = module.AsyncProcessingDialog(None, owner)
            # the 30 s auto refresh would poll at a wall-clock moment; the steps poll explicitly
            self.obj.refresh_timer.stop()

    @property
    def jobs_file(self):
        return self.obj.processor.jobs_file

    def pump(self):
        if self.app is None:
            return
        for _ in range(30):
            self.app.processEvents()
            time.sleep(0.005)

    def select(self, job_ids):
        if self.kind == "headless":
            self.obj.selected_job_ids = list(job_ids)
            self.obj.selected_job_id = job_ids[0] if job_ids else None
            return
        tree = self.obj.jobs_tree
        tree.clearSelection()
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            item.setSelected(item.data(0, self.module.Qt.UserRole) in job_ids)
        self.obj.selected_job_id = job_ids[0] if job_ids else None

    def call(self, method):
        os.environ["OUTPUT_DIRECTORY"] = str(self.out_root)
        getattr(self.obj, method)()
        thread = getattr(self.obj, "processing_thread", None)
        if method == "_start_processing" and thread is not None:
            thread.join(timeout=120)
            self.obj.processing_thread = None
        self.pump()

    def close(self):
        if self.kind == "headless":
            return
        with contextlib.suppress(Exception):
            self.obj.refresh_timer.stop()
        with contextlib.suppress(Exception):
            self.obj.dialog.hide()
            self.obj.dialog.deleteLater()
        self.pump()


def drive(side, epub):
    """The scripted session; returns the observables after every step."""
    owner = side.owner
    steps = []

    def snap(label):
        roots = (side.root, epub.parent)
        steps.append({"step": label, "jobs": jobs_snapshot(side.jobs_file, roots),
                      "trees": {p.name: tree_snapshot(p, roots) for p in sorted(side.out_root.glob("*"))},
                      "calls": norm(side.providers.calls, roots)})

    def ids_for(provider):
        return [jid for jid, job in side.obj.processor.jobs.items() if job.provider == provider]

    owner.file_path = str(epub)
    owner.api_key_entry.setText(SECRET)
    for model in ("gpt-4o-mini", "claude-sonnet-4-5", "gemini-2.5-flash", "mistral-large-latest",
                  "groq/llama-3.1-8b-instant"):
        owner.model_var = model
        side.call("_start_processing")
        snap(f"submit {model}")
    owner.model_var = "gpt-4o-mini"
    side.call("_estimate_cost")
    snap("estimate")
    for provider in ("openai", "anthropic", "gemini"):
        side.select(ids_for(provider)[:1])
        side.call("_check_selected_status")
        snap(f"status {provider}")
    side.select(ids_for("openai")[:1])
    side.call("_retrieve_selected_results")
    snap("retrieve openai")
    side.recorder_answers({"Directory Exists": "no"})
    side.select(ids_for("openai")[:1])
    side.call("_retrieve_selected_results")
    snap("retrieve openai again (create new)")
    side.recorder_answers({"Directory Exists": "yes"})
    side.select(ids_for("anthropic"))
    side.call("_retrieve_selected_results")
    snap("retrieve anthropic (overwrite)")
    side.select(ids_for("gemini"))
    side.call("_retrieve_selected_results")
    snap("retrieve gemini")
    side.select(ids_for("mistral") + ids_for("anthropic"))
    side.call("_cancel_selected_job")
    snap("cancel")
    side.select(ids_for("gemini"))
    side.call("_delete_selected_job")
    snap("delete")
    side.call("_clear_completed_jobs")
    snap("clear")
    owner.model_var = "deepseek-chat"
    side.call("_start_processing")
    owner.model_var = "authgpt/gpt-5"
    owner.api_key_entry.setText("oauth-token-not-a-secret-key")
    side.call("_start_processing")
    snap("unsupported + authgpt")
    owner.file_path = None
    side.call("_estimate_cost")
    side.call("_start_processing")
    snap("no file")
    return steps


def _make_side(name, kind, module, root, owner, app, providers, monkeypatch, recorder):
    side = Side(name, root, owner, kind, module, app=app, providers=providers)
    if kind == "headless":
        def recorder_answers(extra):
            side.obj.answers = dict(ANSWERS, **extra)
    else:
        def recorder_answers(extra):
            recorder.answers = dict(ANSWERS, **extra)
    side.recorder_answers = recorder_answers
    return side


def _run_session(name, kind, module, base, app, monkeypatch, recorder, epub):
    root = base / name
    root.mkdir(parents=True, exist_ok=True)
    providers = FakeProviders()
    monkeypatch.setattr(requests, "post", providers.post)
    monkeypatch.setattr(requests, "get", providers.get)
    monkeypatch.setattr(requests, "delete", providers.delete)
    sdk = []
    install_fake_genai(monkeypatch, sdk)
    logs = []
    from headless_owner import HeadlessOwner

    host = types.SimpleNamespace(log=lambda m: logs.append(str(m)), emit=lambda *a, **k: None,
                                 is_stop_requested=lambda: False)
    owner = HeadlessOwner({"model": "gpt-4o-mini"}, host=host, api_key=SECRET)
    if kind == "dialog":
        monkeypatch.setattr(core, "__file__", str(root / "async_batch_core.py"))
    side = _make_side(name, kind, module, root, owner, app, providers, monkeypatch, recorder)
    if recorder is not None:
        recorder.events = []
        recorder.answers = dict(ANSWERS)
    try:
        steps = drive(side, epub)
    finally:
        side.close()
    roots = (root, epub.parent)
    boxes = norm(recorder.events, roots) if recorder is not None else norm(
        [[m["level"], m["title"], m["text"], m.get("answer")] for m in side.obj.messages], roots)
    return {"steps": steps, "boxes": boxes, "logs": collections.Counter(norm(logs, roots)), "sdk": norm(sdk, roots)}


@pytest.fixture
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_dialog_workflow_matches_the_frozen_dialog_and_headless_reproduces_it(qapp, tmp_path, monkeypatch, capsys):
    import async_api_processor
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "library"))
    for key in ("ENABLE_GEMINI_THINKING", "GEMINI_THINKING_LEVEL", "THINKING_BUDGET", "DISABLE_GEMINI_SAFETY"):
        monkeypatch.delenv(key, raising=False)
    recorder = BoxRecorder(QMessageBox)
    recorder.install(monkeypatch)
    with headless_owner(tmp_path / "sandbox", monkeypatch, {"model": "gpt-4o-mini"}):
        books = tmp_path / "books"
        books.mkdir()
        epub = books / "Book.epub"
        build_epub(epub)
        legacy = load_legacy_module(tmp_path / "legacy" / "async_api_processor.py")
        frozen = _run_session("legacy", "dialog", legacy, tmp_path / "legacy", qapp, monkeypatch, recorder, epub)
        rewired = _run_session("rewired", "dialog", async_api_processor, tmp_path / "rewired", qapp, monkeypatch, recorder, epub)
        headless = _run_session("headless", "headless", core, tmp_path / "headless", None, monkeypatch, None, epub)

    assert [s["step"] for s in frozen["steps"]] == [s["step"] for s in rewired["steps"]]
    for old, new in zip(frozen["steps"], rewired["steps"]):
        assert old == new, old["step"]
    assert frozen["boxes"] == rewired["boxes"]
    assert frozen["logs"] == rewired["logs"]
    assert frozen["sdk"] == rewired["sdk"]
    # the session reached every branch it scripts
    titles = [b[1] for b in rewired["boxes"]]
    for title in ("Start Async Processing", "Gemini Batch API", "Job Status", "Async Translation Complete",
                  "Directory Exists", "Cancel Jobs", "Confirm Delete", "Clear Completed Jobs", "Possibly Unsupported",
                  "No File"):
        assert title in titles, title
    final = rewired["steps"][-1]
    assert sorted(final["trees"]) == ["Book", "Book_1"]
    assert {"c1.xhtml", "c2.xhtml", "c3.xhtml", "translation_progress.json", "metadata.json",
            "chapters_info.json", "images/pic.png"} <= set(final["trees"]["Book"])
    # mobile reproduces the desktop files, provider calls and SDK calls step by step
    for desk, mob in zip(rewired["steps"], headless["steps"]):
        assert desk == mob, desk["step"]
    assert len(headless["steps"]) == len(rewired["steps"])
    assert rewired["sdk"] == headless["sdk"]
    mobile_titles = [b[1] for b in headless["boxes"]]
    for title in ("Job Status", "Async Translation Complete", "Cancel Jobs", "Confirm Delete", "Clear Completed Jobs",
                  "Possibly Unsupported", "Batch Submitted"):
        assert title in mobile_titles, title


# =============================================================================================
# tier H: headless semantics
# =============================================================================================

class _Host:
    def __init__(self, answers=None):
        self.logs, self.events, self.asked = [], [], []
        self.answers = dict(answers or {})
        self.stop = threading.Event()

    def log(self, text, **kw):
        self.logs.append(str(text))

    def emit(self, kind, **data):
        self.events.append((kind, data))

    def ask(self, kind, **data):
        self.asked.append((kind, data))
        return self.answers.get(data.get("title"), data.get("default"))

    def is_stop_requested(self):
        return self.stop.is_set()

    def is_graceful_stop(self):
        return False


def test_headless_questions_go_to_the_host_and_presets_win(isolated, tmp_path, monkeypatch, capsys):
    host = _Host({"Start Async Processing": "no"})
    with headless_owner(tmp_path / "s", monkeypatch, {"model": "gpt-4o-mini"}, host=host, api_key=SECRET) as owner:
        epub = tmp_path / "Book.epub"
        build_epub(epub)
        owner.file_path = str(epub)
        batch = core.HeadlessAsyncBatch(owner, host=host, jobs_file=str(tmp_path / "jobs.json"))
        assert batch.model_status() == {"model": "gpt-4o-mini", "supported": True, "text": "✓ Supported (OPENAI)"}
        assert batch.submit() is None and isolated.calls == []
        kind, data = host.asked[-1]
        assert kind == "async_batch_question" and data["title"] == "Start Async Processing"
        assert data["buttons"] == ["yes", "no"] and data["default"] == "no" and data["level"] == "question"
        batch.answers = {"Start Async Processing": "yes"}
        job = batch.submit()
        assert job is not None and job.job_id == "batch_2" and job.metadata["source_file"] == str(epub)
        assert len(host.asked) == 1  # the preset answered the second time
        assert ("async_batch_message", {"level": "information", "title": "Batch Submitted",
                                        "text": next(m["text"] for m in batch.messages if m["title"] == "Batch Submitted")}) in host.events
        assert json.loads((tmp_path / "jobs.json").read_text(encoding="utf-8"))["batch_2"]["status"] == "pending"
        rows = batch.rows()
        assert rows[0]["job_id"] == "batch_2" and rows[0]["source_file"] == "Book.epub" and rows[0]["progress"] == "0% (Waiting)"
        assert core.selected_job_progress(job) == (0, "0% (0/3 chapters)")
        # no host, no preset: questions answer "no" (nothing destructive happens unasked)
        bare = core.HeadlessAsyncBatch(owner, jobs_file=str(tmp_path / "jobs.json"))
        bare.delete(["batch_2"])
        assert "batch_2" in bare.jobs and bare.messages[-1]["answer"] == "no"


def test_headless_wait_for_completion_polls_without_stack_growth_and_honours_stop(isolated, tmp_path, monkeypatch, capsys):
    host = _Host({"Start Async Processing": "yes"})
    depths = []
    polls = {"n": 0}
    real_get = isolated.get

    def counting_get(url, headers=None, **kw):
        if re.match(r"https://api\.openai\.com/v1/batches/batch_\d+$", url):
            polls["n"] += 1
            depths.append(len(inspect.stack(0)))
            isolated.openai_state = "in_progress" if polls["n"] < 4 else "completed"
        return real_get(url, headers=headers, **kw)

    monkeypatch.setattr(requests, "get", counting_get)
    with headless_owner(tmp_path / "s", monkeypatch, {"model": "gpt-4o-mini", "async_wait_for_completion": True},
                        host=host, api_key=SECRET) as owner:
        epub = tmp_path / "Book.epub"
        build_epub(epub)
        owner.file_path = str(epub)
        monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "out"))
        batch = core.HeadlessAsyncBatch(owner, host=host, jobs_file=str(tmp_path / "jobs.json"))
        assert batch._async_poll_interval() == 60  # the dialog's spin box default, within 10-600
        monkeypatch.setattr(batch, "_async_poll_interval", lambda: 0.001)
        job = batch.submit()
        # the first poll runs inside _start_polling; every later one from the same queue frame
        assert polls["n"] == 4 and len(set(depths[1:])) == 1
        assert batch.jobs[job.job_id].status == core.AsyncAPIStatus.COMPLETED
        assert (tmp_path / "out" / "Book" / "translation_progress.json").is_file()
        # a set stop latch ends the wait before the next poll
        isolated.openai_state = "in_progress"
        polls["n"] = 0
        host.stop.set()
        batch.submit()
        assert polls["n"] == 1
        assert "Async polling stopped" in capsys.readouterr().out


def test_headless_retrieve_reports_output_folders(isolated, tmp_path, monkeypatch, capsys):
    host = _Host({"Start Async Processing": "yes", "Directory Exists": "no"})
    with headless_owner(tmp_path / "s", monkeypatch, {"model": "claude-sonnet-4-5"}, host=host, api_key=SECRET) as owner:
        epub = tmp_path / "Book.epub"
        build_epub(epub)
        owner.file_path = str(epub)
        monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "out"))
        batch = core.HeadlessAsyncBatch(owner, host=host, jobs_file=str(tmp_path / "jobs.json"))
        job = batch.submit()
        assert batch.refresh_statuses()[0]["status"] == "Completed"
        first = batch.retrieve([job.job_id])
        second = batch.retrieve([job.job_id])
        assert first == [str(tmp_path / "out" / "Book")] and second == [str(tmp_path / "out" / "Book_1")]
        progress = json.loads((tmp_path / "out" / "Book" / "translation_progress.json").read_text(encoding="utf-8"))
        assert progress["async_translated"] is True and progress["completed_chapters"] == 3
        assert sorted(v["output_file"] for v in progress["chapters"].values()) == ["c1.xhtml", "c2.xhtml", "c3.xhtml"]
        html = (tmp_path / "out" / "Book" / "c2.xhtml").read_text(encoding="utf-8")
        assert "Claude 0001_c2" in html
