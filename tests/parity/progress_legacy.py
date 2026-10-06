"""U5 parity oracle for the Progress Manager / Glossary Progress extraction.

Freezes ``src/Retranslation_GUI.py`` and ``TransateKRtoEN.ProgressManager.cleanup_missing_files``
at the parent commit of the U5 move (``BASE_SHA``) and drives the legacy and the
working-tree Progress Manager the same way:

* ``legacy_rg()`` / ``current_rg()``: the frozen and the live Retranslation_GUI modules
  (the frozen one cleans up through the frozen TK method, never the live delegate);
* ``block_function(...)``: a legacy statement range executed as a function (the
  verbatim legacy code as an oracle for split / parameterised moves);
* ``make_host`` / ``open_progress_manager`` / ``view_snapshot`` / ``context_action``:
  an offscreen PM dialog, its rows (text, colour, hidden, status), its statistics
  labels, and context-menu actions answered automatically;
* fixture workspaces (EPUB with chunks / metadata / artifacts / QA rows, PDF bookmark
  sections, subtitle ZIP, plain text, image folder) and ``tree_snapshot`` (output tree +
  progress JSON with generated timestamps normalised).

Usage::

    python tests/parity/progress_legacy.py --freeze [--sha REV]
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import types
import zipfile
from pathlib import Path

PARITY_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARITY_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
LEGACY_DIR = PARITY_DIR / "legacy_progress"

#: Parent commit of the U5 progress extraction (the merged U4 commit).
BASE_SHA = "20b446b06b10e60195769dd9c2c9078249300a9c"
SHA12 = BASE_SHA[:12]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

#: Keys whose values the code under test stamps with ``time.time()``.
GENERATED_TIME_KEYS = frozenset({"last_updated", "tts_at", "completed_at", "updated_at", "refined_at", "qa_timestamp"})
#: Fixture timestamps stay below this; anything later was generated during the run.
FIXTURE_EPOCH_LIMIT = 1_500_000_000


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------


def _git_show(path, sha=BASE_SHA):
    raw = subprocess.check_output(["git", "show", f"{sha}:{path}"], cwd=str(REPO_ROOT))
    return raw.decode("utf-8")


def frozen_dir(sha=BASE_SHA):
    return LEGACY_DIR / sha[:12]


def freeze(sha=BASE_SHA):
    """Write the frozen RG module and the TK cleanup method source."""
    target = frozen_dir(sha)
    target.mkdir(parents=True, exist_ok=True)
    rg = _git_show("src/Retranslation_GUI.py", sha)
    (target / "Retranslation_GUI.py").write_text(rg, encoding="utf-8", newline="")
    tk = _git_show("src/TransateKRtoEN.py", sha).replace("\r\n", "\n")
    tree = ast.parse(tk)
    method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ProgressManager":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "cleanup_missing_files":
                    method = item
    assert method is not None
    lines = tk.split("\n")
    body = textwrap.dedent("\n".join(lines[method.lineno - 1:method.end_lineno]))
    (target / "TK_cleanup_missing_files.py").write_text(body + "\n", encoding="utf-8")
    manifest = {
        "sha": sha,
        "Retranslation_GUI.py": hashlib.sha256(rg.encode("utf-8")).hexdigest(),
    }
    (target / "MANIFEST.json.txt").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def _ensure_frozen(sha=BASE_SHA):
    target = frozen_dir(sha)
    if not (target / "Retranslation_GUI.py").exists() or not (target / "TK_cleanup_missing_files.py").exists():
        freeze(sha)
    return target


def legacy_source_lines(sha=BASE_SHA):
    """Frozen RG source lines (LF, 1-based access via ``lines[n - 1]``)."""
    text = (_ensure_frozen(sha) / "Retranslation_GUI.py").read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").split("\n")


# ---------------------------------------------------------------------------
# Legacy cleanup + modules
# ---------------------------------------------------------------------------


def _legacy_cleanup_function(sha=BASE_SHA):
    source = (_ensure_frozen(sha) / "TK_cleanup_missing_files.py").read_text(encoding="utf-8")
    from metadata_progress import is_metadata_progress_entry

    namespace = {"os": os, "time": time, "is_metadata_progress_entry": is_metadata_progress_entry}
    exec(compile(source, f"<legacy TK cleanup {sha[:12]}>", "exec"), namespace)
    return namespace["cleanup_missing_files"]


class LegacyProgressManager:
    """Stand-in for TransateKRtoEN.ProgressManager with the frozen cleanup method."""

    _cleanup = None

    def __init__(self, payloads_dir):
        self.payloads_dir = payloads_dir
        self.prog = {}

    def cleanup_missing_files(self, output_dir):
        if LegacyProgressManager._cleanup is None:
            LegacyProgressManager._cleanup = _legacy_cleanup_function()
        return LegacyProgressManager._cleanup(self, output_dir)


def legacy_cleanup_missing_files(prog, output_dir):
    manager = LegacyProgressManager("")
    manager.prog = prog
    manager.cleanup_missing_files(output_dir)
    return manager.prog


_MODULES = {}


def _neutralise(module, legacy):
    module._schedule_epub_reader_engine_prewarm = lambda: None
    if legacy:
        module._get_progress_manager_nonblocking = lambda: LegacyProgressManager
    else:
        module._get_progress_manager_nonblocking = lambda: LegacyProgressManager


def legacy_rg(sha=BASE_SHA):
    """The frozen Retranslation_GUI module (imported once per process)."""
    key = ("legacy", sha)
    if key not in _MODULES:
        path = _ensure_frozen(sha) / "Retranslation_GUI.py"
        name = f"legacy_progress_rg_{sha[:12]}"
        spec = importlib.util.spec_from_file_location(name, str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _neutralise(module, legacy=True)
        _MODULES[key] = module
    return _MODULES[key]


def current_rg():
    """The working-tree Retranslation_GUI module."""
    key = ("current", None)
    if key not in _MODULES:
        import Retranslation_GUI as module

        _neutralise(module, legacy=False)
        _MODULES[key] = module
    return _MODULES[key]


# ---------------------------------------------------------------------------
# Legacy statement ranges as functions
# ---------------------------------------------------------------------------


def block_function(module, start, end, params, *, dedent, result="locals", name="legacy_block",
                   sha=BASE_SHA, prologue=""):
    """Execute frozen RG lines ``start..end`` (dedented) as ``def name(*params)``.

    The function runs in ``module``'s globals (the frozen module) and returns its
    ``locals()`` (``result='locals'``) or the expression ``result``.
    """
    lines = legacy_source_lines(sha)[start - 1:end]
    body = []
    for line in lines:
        if line.strip():
            assert line.startswith(" " * dedent), (start, end, line)
            body.append("    " + line[dedent:])
        else:
            body.append("")
    if prologue:
        body = ["    " + line for line in textwrap.dedent(prologue).strip("\n").split("\n")] + body
    ret = "    return dict(locals())" if result == "locals" else f"    return {result}"
    source = f"def {name}({', '.join(params)}):\n" + "\n".join(body) + "\n" + ret + "\n"
    namespace = dict(vars(module))
    exec(compile(source, f"<{name} RG {start}-{end}>", "exec"), namespace)
    return namespace[name]


# ---------------------------------------------------------------------------
# Normalisation and snapshots
# ---------------------------------------------------------------------------


def normalize_progress(value):
    """Replace run-generated timestamps (>= FIXTURE_EPOCH_LIMIT) with '<now>'."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if (
                key in GENERATED_TIME_KEYS
                and isinstance(item, (int, float))
                and not isinstance(item, bool)
                and item >= FIXTURE_EPOCH_LIMIT
            ):
                out[key] = "<now>"
            else:
                out[key] = normalize_progress(item)
        return out
    if isinstance(value, list):
        return [normalize_progress(item) for item in value]
    return value


def _replace_paths(value, replacements):
    if isinstance(value, dict):
        return {key: _replace_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_paths(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_paths(item, replacements) for item in value)
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    return value


def tree_snapshot(root, *, rel_to=None, workspace=None):
    """``{relative path: content}`` of a folder; JSON files parsed + normalised.

    ``workspace`` (the copy's root folder) is replaced by ``<ROOT>`` in text content,
    so two copies of one fixture compare equal.
    """
    root = Path(root)
    replacements = []
    if workspace is not None:
        ws = str(Path(workspace))
        replacements = [(ws, "<ROOT>"), (ws.replace("\\", "/"), "<ROOT>")]
    out = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(rel_to or root).as_posix()
        if rel.endswith(".lock"):
            continue
        data = path.read_bytes()
        if path.suffix == ".json":
            try:
                parsed = json.loads(data.decode("utf-8"))
                out[rel] = ("json", _replace_paths(normalize_progress(parsed), replacements))
                continue
            except ValueError:
                pass
        if replacements and len(data) < 65536:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = None
            if text is not None:
                out[rel] = ("text", _replace_paths(text, replacements))
                continue
        out[rel] = ("bytes", hashlib.sha256(data).hexdigest())
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


OPF_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Fixture</dc:title><dc:language>ko</dc:language></metadata>
<manifest>
{manifest}
</manifest>
<spine>
{spine}
</spine>
</package>
"""


def make_epub(path, chapters):
    """Write a minimal EPUB: ``chapters`` = [(filename, html), ...] in spine order."""
    manifest = []
    spine = []
    for index, (filename, _html) in enumerate(chapters):
        manifest.append(f'<item id="c{index}" href="Text/{filename}" media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="c{index}"/>')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        archive.writestr("OEBPS/content.opf", OPF_TEMPLATE.format(manifest="\n".join(manifest), spine="\n".join(spine)))
        for filename, html in chapters:
            archive.writestr(f"OEBPS/Text/{filename}", html)


def _html(body):
    return f"<html><head><title>t</title></head><body>{body}</body></html>"


def _write(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def _set_mtime(root, stamp=1_000_000.0):
    for path in Path(root).rglob("*"):
        try:
            os.utime(path, (stamp, stamp))
        except OSError:
            pass


def _chunk_marked(chapter_key, total, contents):
    import chapter_chunk_progress as ccp

    key = ccp.chunk_marker_key(chapter_key) if hasattr(ccp, "chunk_marker_key") else hashlib.sha256(
        str(chapter_key).encode("utf-8")).hexdigest()[:16]
    parts = []
    for index, content in enumerate(contents, start=1):
        parts.append(f"<!-- GLOSSARION_CHUNK_START key={key} idx={index} total={total} -->")
        parts.append(content)
        parts.append(f"<!-- GLOSSARION_CHUNK_END key={key} idx={index} -->")
    return "\n".join(parts)


def epub_workspace(base):
    """EPUB source + output folder exercising every PM row kind and action."""
    base = Path(base)
    source = base / "Book.epub"
    names = [
        "title.xhtml", "chapter0001.xhtml", "chapter0002.xhtml", "chapter0003.xhtml",
        "chapter0004.xhtml", "chapter0005.xhtml", "chapter0006.xhtml", "chapter0007.xhtml",
        "chapter0008.xhtml", "notice.xhtml",
    ]
    make_epub(source, [
        (name, _html(f"<p>원문 {name}</p><img src='../Images/i{index}.png'/>"))
        for index, name in enumerate(names)
    ])
    out = base / "out" / "Book"
    out.mkdir(parents=True)
    chunk_key = "hash-ch3"
    _write(out / "response_chapter0001.html", _html("<p>Chapter one</p>"))
    _write(out / "response_chapter0002.html", _html("<p>Chapter two <b x=''>llm</b></p>"))
    _write(out / "response_chapter0003.html", _html(_chunk_marked(chunk_key, 3, ["<p>a</p>", "<p>b</p>", "<p>c</p>"])))
    _write(out / "response_chapter0004.html", _html("<p>Chapter four pending html</p>"))
    _write(out / "response_chapter0005.html", _html("<p>Chapter five previous</p>"))
    _write(out / "response_chapter0007.html", _html("<p>Chapter seven untracked</p>"))
    _write(out / "text_to_speech" / "response_chapter0001.mp3", "audio")
    _write(out / "metadata.json", json.dumps({"title": "책", "title_translated": True, "creator": "작가"}))
    _write(out / "TOC.txt", "Translated: Contents\n")
    prog = {
        "version": "2.1",
        "chapters": {
            "1": {"actual_num": 1, "content_hash": "h1", "output_file": "response_chapter0001.html",
                  "status": "completed", "last_updated": 1000.0, "original_basename": "chapter0001.xhtml",
                  "model_name": "gpt-x", "tts_status": "tts_completed",
                  "tts_file": "text_to_speech/response_chapter0001.mp3", "refinement_status": "refined"},
            "2": {"actual_num": 2, "content_hash": "h2", "output_file": "response_chapter0002.html",
                  "status": "qa_failed", "last_updated": 1001.0, "original_basename": "chapter0002.xhtml",
                  "model_name": "gpt-x", "qa_issues": True,
                  "qa_issues_found": ["llm_token_issue_empty_attr", "missing_images_1",
                                      "korean_text_found_12_chars_"],
                  "qa_issue_previews": {"llm_token_issue_empty_attr": "<b x=''>"},
                  "qa_timestamp": 1001.5},
            "3": {"actual_num": 3, "content_hash": chunk_key, "output_file": "response_chapter0003.html",
                  "status": "completed", "last_updated": 1002.0, "original_basename": "chapter0003.xhtml",
                  "model_name": "gpt-y"},
            "4": {"actual_num": 4, "content_hash": "h4", "output_file": "response_chapter0004.html",
                  "status": "pending", "last_updated": 1003.0, "original_basename": "chapter0004.xhtml",
                  "previous_progress_entry": {"status": "completed", "model_name": "gpt-z"}},
            "5": {"actual_num": 5, "content_hash": "h5", "output_file": "response_chapter0005.html",
                  "status": "in_progress", "last_updated": 1004.0, "original_basename": "chapter0005.xhtml",
                  "previous_status": "completed",
                  "previous_progress_entry": {"status": "completed", "model_name": "gpt-old",
                                              "output_file": "response_chapter0005.html",
                                              "refinement_status": "failed"}},
            "6": {"actual_num": 6, "content_hash": "h6", "output_file": "response_chapter0006.html",
                  "status": "in_progress", "last_updated": 1005.0, "original_basename": "chapter0006.xhtml",
                  "previous_status": "not_translated"},
            "8": {"actual_num": 8, "content_hash": "h8", "output_file": "response_chapter0008.html",
                  "status": "completed", "last_updated": 1006.0, "original_basename": "chapter0008.xhtml"},
            "9": {"actual_num": 9, "content_hash": "h9", "output_file": "response_chapter0001.html",
                  "status": "merged", "merged_parent_chapter": 1, "last_updated": 1007.0,
                  "original_basename": "chapter0009.xhtml"},
        },
        "chapter_chunks": {
            chunk_key: {
                "schema_version": 2, "total": 3, "completed": [1, 2, 3],
                "chunks": {"1": "<p>a</p>", "2": "<p>b</p>", "3": "<p>c</p>"},
                "chunk_metadata": {},
                "entries": {
                    "1": {"index": 1, "status": "completed"},
                    "2": {"index": 2, "status": "qa_failed", "qa_issues_found": ["korean_text_found_3_chars_"]},
                    "3": {"index": 3, "status": "completed"},
                },
                "chapter_status": "qa_failed",
            }
        },
    }
    _write(out / "translation_progress.json", json.dumps(prog, ensure_ascii=False, indent=2))
    _set_mtime(base)
    config = {
        "output_directory": str(base / "out"),
        "translate_book_title": True,
        "metadata_translation_mode": "together",
        "use_toc_ncx": True,
        "batch_translate_headers": False,
        "special_file_keywords": "title, notice",
        "special_file_exact": "index",
        "translate_special_files": False,
        "translate_all_numbered_html": True,
        "retranslation_show_model_info": False,
    }
    return source, out, config


def pdf_workspace(base):
    """PDF source with three bookmark sections (plan injected; no PDF parsing)."""
    base = Path(base)
    source = base / "Manual.pdf"
    source.write_bytes(b"%PDF-1.4 fixture")
    out = base / "out" / "Manual"
    out.mkdir(parents=True)
    _write(out / "response_pdf_section_1.html", _html("<p>s1</p>"))
    prog = {"version": "2.1", "chapters": {
        "1": {"actual_num": 1, "content_hash": "p1", "output_file": "response_pdf_section_1.html",
              "status": "completed", "last_updated": 1000.0, "original_basename": "pdf_section_1.html"},
    }, "chapter_chunks": {}}
    _write(out / "translation_progress.json", json.dumps(prog, indent=2))
    _set_mtime(base)
    plan = [
        {"num": 1, "title": "Intro", "start_page": 1, "end_page": 3, "level": 0},
        {"num": 2, "title": "Setup", "start_page": 4, "end_page": 4, "level": 1},
        {"num": 3, "title": "Usage", "start_page": 5, "end_page": 9, "level": 0, "section_id": "s3"},
    ]
    config = {"output_directory": str(base / "out"), "translate_book_title": False,
              "pdf_use_toc_sections": True, "use_toc_ncx": False, "batch_translate_headers": False}
    return source, out, config, plan


def subtitle_zip_workspace(base):
    base = Path(base)
    source = base / "Season.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("ep01.srt", "1\n00:00:01,000 --> 00:00:02,000\n안녕\n")
        archive.writestr("ep02.srt", "1\n00:00:01,000 --> 00:00:02,000\n잘가\n")
    out = base / "out" / "Season"
    out.mkdir(parents=True)
    _write(out / "ep01.srt", "1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    _set_mtime(base)
    config = {"output_directory": str(base / "out")}
    return source, out, config


def text_workspace(base):
    base = Path(base)
    source = base / "Story.txt"
    source.write_text("원문\n", encoding="utf-8")
    out = base / "out" / "Story"
    out.mkdir(parents=True)
    _write(out / "response_section_1.txt", "one")
    _write(out / "response_section_2.txt", "two")
    _write(out / "response_section_3.txt", "three")
    prog = {"version": "2.1", "chapters": {
        "1": {"actual_num": 1, "content_hash": "t1", "output_file": "response_section_1.txt",
              "status": "completed", "last_updated": 1000.0},
        "2": {"actual_num": 2, "content_hash": "t2", "output_file": "response_section_2.txt",
              "status": "failed", "last_updated": 1001.0, "failure_reason": "boom"},
        "4": {"actual_num": 4, "content_hash": "t4", "output_file": "response_section_4.txt",
              "status": "completed", "last_updated": 1002.0},
    }, "chapter_chunks": {}}
    _write(out / "translation_progress.json", json.dumps(prog, indent=2))
    _set_mtime(base)
    config = {"output_directory": str(base / "out")}
    return source, out, config


def image_folder_workspace(base):
    base = Path(base)
    folder = base / "Pics"
    folder.mkdir(parents=True)
    for name in ("a.png", "b.png"):
        (folder / name).write_bytes(b"\x89PNG fixture")
    out = base / "Pics_translated"
    out.mkdir(parents=True)
    _write(out / "response_001_a.html", _html("<p>a</p>"))
    (out / "images").mkdir()
    (out / "images" / "cover.png").write_bytes(b"\x89PNG")
    _set_mtime(base)
    config = {"output_directory": ""}
    return folder, out, config


GLOSSARY_CSV = "type,raw_name,translated_name,gender,description\ncharacter,김철수,Kim Cheolsu,male,\nterm,마나,Mana,,\n"


def glossary_progress_fixture(base, source):
    """``Glossary/<book>/<book>_glossary_progress.json`` + glossary CSV next to it."""
    base = Path(base)
    book = Path(source).stem
    gdir = base / "out" / "Glossary" / book
    gdir.mkdir(parents=True, exist_ok=True)
    _write(gdir / f"{book}_glossary.csv", GLOSSARY_CSV)
    data = {
        "book_title": "Fixture Book",
        "progress_schema_version": "2.2",
        "indexing": "chapter_index_zero_based",
        "chapter_count": 7,
        "chapter_filenames": {str(i): f"chapter{i + 1:04d}.xhtml" for i in range(7)},
        "completed": [0, 1],
        "skipped": [3],
        "failed": [2],
        "in_progress": [4],
        "merged_indices": [],
        "qa_issues_found": {"2": ["entry_count_low"]},
        "chapters": {
            "0": {"chapter_index": 0, "actual_num": 1, "status": "completed", "model_name": "g-1",
                  "output_file": "chapter0001.xhtml", "last_updated": 1000.0},
            "2": {"chapter_index": 2, "actual_num": 3, "status": "qa_failed", "model_name": "g-1",
                  "qa_issues_found": ["entry_count_low"], "output_file": "chapter0003.xhtml",
                  "last_updated": 1001.0},
            "5": {"chapter_index": 5, "actual_num": 6, "status": "skipped_image_only", "model_name": "SKIPPED",
                  "output_file": "chapter0006.xhtml", "last_updated": 1002.0},
        },
        "refinement": {
            "type::character": {"entry_type": "character", "status": "completed", "model_name": "g-2",
                                "entry_count_before": 1, "entry_count_after": 1, "last_updated": 1003.0},
        },
        "minimal_pass": {"status": "completed", "model_name": "g-3", "entry_count": 2, "updated_at": 1004.0},
        "progress_session_id": "sess-1",
    }
    path = gdir / f"{book}_glossary_progress.json"
    _write(path, json.dumps(data, ensure_ascii=False, indent=2))
    _set_mtime(gdir)
    return path


# ---------------------------------------------------------------------------
# Offscreen Progress Manager harness
# ---------------------------------------------------------------------------

#: Every ``_styled_msgbox`` / ``_show_message`` call: (kind, title, text).
MESSAGES = []


def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def pump(ms=50, until=None, timeout=5.0):
    from PySide6.QtCore import QEventLoop
    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + timeout
    while True:
        QApplication.processEvents(QEventLoop.AllEvents, ms)
        time.sleep(ms / 1000.0)
        if until is None:
            if time.monotonic() >= deadline - timeout + ms / 1000.0:
                return True
        elif until():
            return True
        if time.monotonic() > deadline:
            return False


_HOSTS = {}


def make_host(module, config, **attrs):
    """A QWidget + RetranslationMixin owner wired like the desktop owner (special-file
    rules from the translation pipeline, vars seeded from ``config``)."""
    qapp()
    from PySide6.QtWidgets import QMessageBox, QWidget
    from owner_state import ConfigStateMixin
    from run_env import RunEnvMixin
    from settings_rules import _config_var
    from translation_pipeline import GlossaryPipelineMixin

    key = id(module)
    if key not in _HOSTS:
        class Host(QWidget, module.RetranslationMixin):
            _is_special_file = GlossaryPipelineMixin._is_special_file
            _should_skip_special_file = GlossaryPipelineMixin._should_skip_special_file
            _get_output_mode = RunEnvMixin._get_output_mode
            _upgrade_special_file_exact = ConfigStateMixin._upgrade_special_file_exact
            _LEGACY_SPECIAL_FILE_EXACT_TOKENS = ConfigStateMixin._LEGACY_SPECIAL_FILE_EXACT_TOKENS

            def __init__(self, config):
                QWidget.__init__(self)
                self.config = config
                self.selected_files = []
                self.saved_configs = 0
                self.logs = []
                for name in ("translate_special_files_var", "special_file_keywords_var",
                             "special_file_exact_var", "translate_all_numbered_html_var"):
                    value = _config_var(config, name)
                    if name == "special_file_exact_var":
                        value = self._upgrade_special_file_exact(value)
                    setattr(self, name, value)

            def save_config(self, show_message=False):
                self.saved_configs += 1

            def append_log(self, message):
                self.logs.append(message)

            def _show_message(self, msg_type, title, message, parent=None):
                MESSAGES.append((msg_type, title, message))
                return True

            @staticmethod
            def _styled_msgbox(icon, parent, title, message, buttons=None):
                MESSAGES.append((str(icon), title, message))
                return QMessageBox.Yes

        _HOSTS[key] = Host
    host = _HOSTS[key](config)
    for name, value in attrs.items():
        setattr(host, name, value)
    return host


def open_progress_manager(host, source, **kwargs):
    """Build the standalone PM dialog, wait for the list, stop the live timers."""
    data = host._force_retranslation_epub_or_text(str(source), **kwargs)
    if data is None:
        return None
    pump(until=lambda: (
        not data.get('_listbox_populate_active')
        and data['listbox'].count() == len(data.get('chapter_display_info') or [])
    ), timeout=10.0)
    settle_silent_refresh(data)
    stop_live_refresh(data)
    return data


def settle_silent_refresh(data, timeout=3.0):
    """Wait for the show-time silent refresh (background snapshot) to be applied."""
    pump(until=lambda: bool(data.get('_last_prefetch_started_at')), timeout=1.0)
    pump(until=lambda: not data.get('_prefetch_running') and not data.get('_prefetch_scheduled'),
         timeout=timeout)
    pump(30, timeout=0.15)
    pump(until=lambda: not data.get('_listbox_populate_active'), timeout=timeout)


def stop_live_refresh(data):
    for key in ('_auto_refresh_timer', '_progress_watch_debounce'):
        timer = data.get(key)
        if timer is not None:
            timer.stop()
    watcher = data.get('_progress_watcher')
    if watcher is not None:
        paths = watcher.files() + watcher.directories()
        if paths:
            watcher.removePaths(paths)


def view_snapshot(data):
    """Rows (text, colour, hidden, status) + statistics labels of a PM view."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel

    listbox = data['listbox']
    rows = []
    for index in range(listbox.count()):
        item = listbox.item(index)
        rows.append((item.text(), item.foreground().color().name(), item.isHidden(), item.data(Qt.UserRole + 2)))
    labels = []
    container = data['container']
    for label in container.findChildren(QLabel):
        text = label.text()
        if text.startswith(('Total:', '✅', '🔗', '🔄', '❓', '⬜', '✨', '🔊', '❌', '💀', '⏭️')):
            labels.append((text, label.isVisibleTo(container), label.styleSheet()))
    return {"rows": rows, "labels": sorted(labels)}


def select_rows(data, predicate):
    listbox = data['listbox']
    listbox.clearSelection()
    first = None
    for index, info in enumerate(data.get('chapter_display_info') or []):
        if predicate(info):
            item = listbox.item(index)
            item.setSelected(True)
            if first is None:
                first = item
    return first


def context_action(module, data, label_prefix, predicate):
    """Select rows matching ``predicate`` and pick the context-menu action ``label_prefix``."""
    from PySide6.QtWidgets import QMenu

    first = select_rows(data, predicate)
    assert first is not None, "no row matched"
    chosen = {}

    class AutoMenu(QMenu):
        def exec(self, *args, **kwargs):
            for action in self.actions():
                if action.text().startswith(label_prefix):
                    chosen['text'] = action.text()
                    return action
            chosen['text'] = None
            return None

    original = module.QMenu
    module.QMenu = AutoMenu
    try:
        listbox = data['listbox']
        listbox.scrollToItem(first)
        pump(20, timeout=0.1)
        rect = listbox.visualItemRect(first)
        listbox.customContextMenuRequested.emit(rect.center())
        pump(20, timeout=0.3)
    finally:
        module.QMenu = original
    return chosen.get('text')


def click_button(data, text):
    from PySide6.QtWidgets import QPushButton

    for button in data['container'].findChildren(QPushButton):
        if button.text().strip() == text:
            button.click()
            pump(20, timeout=0.3)
            return True
    raise AssertionError(f"button {text!r} not found")


def copy_workspace(src, dst):
    shutil.copytree(src, dst, copy_function=shutil.copy2)
    return Path(dst)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--sha", default=BASE_SHA)
    args = parser.parse_args(argv)
    if args.freeze:
        sha = subprocess.check_output(["git", "rev-parse", args.sha], cwd=str(REPO_ROOT)).decode().strip()
        print("frozen", freeze(sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
