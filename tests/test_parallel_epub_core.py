"""U6: parallel_epub_core (the Parallel EPUB Pair mapper without Qt) parity and API tests.

parallel_epub_glossary.py 52-512 moved verbatim into ``parallel_epub_core``; the pure halves
of ``ParallelEpubPairDialog`` methods and of the ``TranslatorGUI`` pair helpers were lifted
into it with explicit parameters, and the dialog / TranslatorGUI keep thin wrappers. The
oracle is the source at ``U6_BASE_SHA`` (``git show``): parallel_epub_glossary.py is loaded
as a separate module and the frozen TranslatorGUI methods are executed against the live
translator_gui globals.

Tiers:
* H (hygiene): parallel_epub_core imports without PySide6 / translator_gui / dpi_setup and
  parses as Python 3.10; line endings stay uniform;
* V (verbatim): the moved block equals the frozen lines, every lifted body equals the frozen
  lines plus the documented edits (``MOVED_BODIES``), parallel_epub_glossary.py differs from
  the frozen file only inside the rewired spans, and the dialog module re-exports the core
  objects;
* D (differential fuzz, >= ``PARITY_U6_STATES`` = 500 states each): the moved functions on
  random chapter sets; the frozen dialog methods vs the working-tree ones on recording fakes
  (offset, unmapping, saved-mapping restore, selected mapping, status line, unpaired counts
  and warning, Use Mapped Pair checks / result, prompt persistence, the background loader);
  the frozen TranslatorGUI pair helpers vs the wrappers (glossary folder, sidecar write /
  read, working EPUB, activated-pair record, saved-pair restore);
* F (file system): fixture EPUB pairs written with ebooklib go through the frozen and the
  working-tree desktop paths (real chapter loader, accept, activate, sidecar, working EPUB,
  restore on launch) and the GUI-free path the mobile job uses; sidecar bytes, the paired
  EPUB's documents and the restored result must match;
* S (smoke): the real ParallelEpubPairDialog, frozen and working-tree, offscreen on fixture
  EPUBs through its background loader, offset and accept.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/test_parallel_epub_core.py
"""

from __future__ import annotations

import ast
import contextlib
import copy
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import textwrap
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

import parallel_epub_core as pec  # noqa: E402

HAS_QT = importlib.util.find_spec("PySide6") is not None
needs_qt = pytest.mark.skipif(not HAS_QT, reason="needs PySide6")

STATES = max(1, int(os.environ.get("PARITY_U6_STATES", "500")))
SEED = int(os.environ.get("PARITY_U6_SEED", "6062"))
#: Real-dialog runs (each builds two offscreen dialogs).
DIALOG_RUNS = max(1, int(os.environ.get("PARITY_U6_DIALOG_RUNS", "6")))

#: U6 base: parallel_epub_glossary.py / translator_gui.py at this commit are the frozen sources.
U6_BASE_SHA = "e28e3a0f8e2ef3bd2aea59c87049b336ea469ac0"

PEG = "src/parallel_epub_glossary.py"
TG = "src/translator_gui.py"

#: new function -> ([(frozen file, first line, last line), ...], edits). The frozen segments
#: are joined, dedented and edited, then compared with the new function's body (its lines
#: after the docstring, dedented). Edits: (old, new) replacements (``old`` must occur),
#: ("re", pattern, repl), ("prepend", text) and ("append", text). Trailing whitespace and
#: leading / trailing blank lines are ignored. Lifted functions that were restructured
#: (offset, saved rows, selected mapping, status, validation, prompt settings) are pinned
#: by the differential tests below instead.
MOVED_BODIES = {
    "load_parallel_epub_documents": ([(PEG, 1145, 1165)], [
        ("self.chapter_loader(path)", "chapter_loader(path)"),
        ("append", "return chapters, reading_order, error"),
    ]),
    "prepare_persisted_parallel_epub_selection": ([(PEG, 1280, 1303)], [
        ("return False\nraw_path", "return None\nraw_path"),
        ("):\n    return False\n", "):\n    return None\n"),
        ("self._pending_persisted_selection = {", "return {"),
    ]),
    "parallel_epub_selection_matches": ([(PEG, 1450, 1461)], [
        ("os.path.abspath(self.raw_path)", "os.path.abspath(raw_path)"),
        ("os.path.abspath(self.translated_path)", "os.path.abspath(translated_path)"),
        ("if current_paths != saved_paths:\n    return False", "return current_paths == saved_paths"),
    ]),
    "translated_mapping_label": ([(PEG, 1536, 1538)], [
        ("re", r"self\.translated_chapters", "translated_chapters"),
    ]),
    "valid_parallel_epub_rows": ([(PEG, 1645, 1651)], [
        ("valid_rows = sorted(", "return sorted("),
        ("self.mapping_table.rowCount()", "row_count"),
    ]),
    "unpaired_file_counts": ([(PEG, 1693, 1696)], [
        ("len(self.raw_chapters)", "raw_count"),
        ("len(self.translated_chapters)", "translated_count"),
    ]),
    "unpaired_warning_text": ([(PEG, 1701, 1712)], [
        ("self._unpaired_file_counts(mapping)",
         "unpaired_file_counts(\n    mapping, raw_count, translated_count\n)"),
    ]),
    "build_parallel_epub_pairs": ([(PEG, 1904, 1917)], [
        ("self.raw_chapters[", "raw_chapters["),
        ("self.translated_chapters[", "translated_chapters["),
        ("append", "return pairs"),
    ]),
    "parallel_epub_profiles": ([(PEG, 795, 799)], [
        ("self.config.get(", "config.get("),
        ("self.profiles = dict(", "profiles = dict("),
        ("re", r"self\.profiles", "profiles"),
        ("append", "return profiles"),
    ]),
    "active_parallel_epub_profile": ([(PEG, 802, 807)], [
        ("self.config.get(", "config.get("),
        ("self.profiles", "profiles"),
        ("append", "return active"),
    ]),
    "load_parallel_epub_chapters": ([(TG, 28460, 28467)], []),
    "build_parallel_epub_pair_artifact": ([(TG, 28477, 28496)], []),
    "resolve_parallel_epub_glossary_output_dir": ([(TG, 20037, 20071)], [
        ("    if not self._parallel_epub_pair_is_selected():\n"
         "        return \"\"\n"
         "    state = getattr(self, \"_parallel_epub_pair_source\", None)\n"
         "    if not isinstance(state, dict):\n"
         "        return \"\"\n"
         "    raw_path = str(state.get(\"raw_path\") or \"\")\n"
         "    if not raw_path:\n"
         "        return \"\"\n", "    return \"\"\n"),
        ("self.config.get(", "config.get("),
    ]),
    "parallel_epub_mapping_sidecar_path": ([(TG, 20078, 20091)], [
        ("folder = self._resolve_parallel_epub_glossary_output_dir(\n    raw_path, create=",
         "folder = resolve_parallel_epub_glossary_output_dir(\n    raw_path, config, create="),
    ]),
    "write_parallel_epub_mapping_sidecar": ([(TG, 20096, 20104)], [
        ("path = self._parallel_epub_mapping_sidecar_path(\n    str(selection.get(\"raw_path\") or \"\"), create_parent",
         "path = parallel_epub_mapping_sidecar_path(\n    str(selection.get(\"raw_path\") or \"\"), config, create_parent"),
        ("_atomic_json_write(path, selection)",
         "from app_paths import _atomic_json_write\n\n_atomic_json_write(path, selection)"),
    ]),
    "read_parallel_epub_mapping_sidecar": ([(TG, 20109, 20128)], [
        ("self._parallel_epub_mapping_sidecar_path(raw_path)", "parallel_epub_mapping_sidecar_path(raw_path, config)"),
    ]),
    "parallel_epub_pair_source_state": ([(TG, 28520, 28525), (TG, 28537, 28550)], [
        ("self._parallel_epub_pair_source = {", "return {"),
    ]),
    "rebuild_parallel_epub_pair_result": ([(TG, 28658, 28696)], [
        ("prepend", "if chapter_loader is None:\n    chapter_loader = load_parallel_epub_chapters\n"
                    "raw_path = str(selection.get(\"raw_path\") or \"\")\n"
                    "translated_path = str(selection.get(\"translated_path\") or \"\")"),
        ("re", r"self\._load_parallel_epub_chapters\(", "chapter_loader("),
        ("re", r"selection_copy", "selection"),
        ("re", r"self\.config", "config"),
        ("append", "return result, skipped"),
    ]),
}

#: names parallel_epub_glossary re-exports from parallel_epub_core
REEXPORTS = (
    "DEFAULT_PARALLEL_EPUB_PROFILE", "DEFAULT_PARALLEL_EPUB_WRAPPER_PROMPT", "PARALLEL_EPUB_SELECTION_CONFIG_KEY",
    "PARALLEL_EPUB_SYSTEM_INSTRUCTIONS", "_chapter_special_flags", "_has_positive_member_number",
    "_member_number_signature", "_nonpositive_member_layout", "_normalized_member_stem",
    "apply_parallel_epub_wrapper", "auto_map_epub_chapters", "chapter_filename", "chapter_text",
    "compact_parallel_epub_selection", "default_parallel_epub_system_prompt", "parallel_epub_working_filename",
    "restore_parallel_epub_pairs", "write_parallel_epub",
)


# =============================================================================================
# frozen sources
# =============================================================================================

_GIT_CACHE = {}


def git_text(relpath, sha=U6_BASE_SHA):
    key = (relpath, sha)
    if key not in _GIT_CACHE:
        try:
            data = subprocess.run(["git", "show", f"{sha}:{relpath}"], cwd=str(REPO_ROOT),
                                  capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            pytest.skip(f"frozen source {relpath}@{sha[:8]} unavailable: {exc}")
        _GIT_CACHE[key] = data.decode("utf-8-sig").replace("\r\n", "\n")
    return _GIT_CACHE[key]


def current_text(name):
    return (SRC / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def _norm_block(text):
    lines = [line.rstrip() for line in textwrap.dedent(text).split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _apply_edits(text, edits, name):
    for edit in edits:
        if edit[0] == "re":
            text = re.sub(edit[1], edit[2], text)
        elif edit[0] == "prepend":
            text = edit[1] + "\n" + text
        elif edit[0] == "append":
            text = text + "\n" + edit[1]
        else:
            old, new = edit
            assert old in text, f"{name}: documented edit not found: {old!r}"
            text = text.replace(old, new)
    return text


def frozen_body(name):
    segments, edits = MOVED_BODIES[name]
    parts = []
    for rel, first, last in segments:
        parts.append(textwrap.dedent("\n".join(
            "" if not line.strip() else line for line in git_text(rel).split("\n")[first - 1:last])))
    return _norm_block(_apply_edits(_norm_block("\n".join(parts)), edits, name))


def function_body(text, name, cls=None):
    """The body of a module-level function (or a class method) after its docstring, dedented."""
    tree = ast.parse(text)
    scope = tree
    if cls is not None:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.split("\n")
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
            and isinstance(first.value.value, str):
        start = first.end_lineno  # 1-based line after the docstring = index end_lineno
    else:
        start = first.lineno - 1
        while start - 1 > node.lineno - 1 and lines[start - 1].strip().startswith("#"):
            start -= 1
    return _norm_block("\n".join(lines[start:node.end_lineno]))


def method_source(text, name, cls="TranslatorGUI"):
    """Dedented source of a class method of a frozen / current file."""
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    meth = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.split("\n")
    start = (meth.decorator_list[0].lineno if meth.decorator_list else meth.lineno) - 1
    return textwrap.dedent("\n".join(lines[start:meth.end_lineno]))


@contextlib.contextmanager
def swapped_module(name, module):
    """``sys.modules[name]`` = module for the duration (the frozen side imports the frozen code)."""
    saved = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def frozen_peg(qapp):
    """parallel_epub_glossary.py at U6_BASE_SHA, imported as a separate module."""
    module = types.ModuleType("_parallel_epub_glossary_u6_frozen")
    module.__file__ = str(SRC / "parallel_epub_glossary.py")
    sys.modules[module.__name__] = module
    exec(compile(git_text(PEG), str(SRC / "parallel_epub_glossary.py"), "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def new_peg(qapp):
    import parallel_epub_glossary
    return parallel_epub_glossary


@pytest.fixture(scope="module")
def tg_module(qapp):
    import translator_gui
    return translator_gui


@pytest.fixture(scope="module")
def frozen_tg(tg_module):
    """Frozen TranslatorGUI methods, executed against the live translator_gui globals."""
    text = git_text(TG)
    cache = {}

    def get(name):
        if name not in cache:
            ns = dict(vars(tg_module))
            exec(compile(method_source(text, name), f"<frozen TranslatorGUI.{name}>", "exec"), ns)
            cache[name] = ns[name]
        return cache[name]

    return get


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Never touch the real Library / output roots; restore the process env and cwd."""
    original = dict(os.environ)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "_library"))
    for key in ("OUTPUT_DIRECTORY", "OUTPUT_DIR", "EPUB_PATH", "GLOSSARY_SHARED_DIR"):
        monkeypatch.delenv(key, raising=False)
    cwd = os.getcwd()
    yield
    # monkeypatch.undo() first: its records may hold values a test set directly (the stop
    # protocol writes TRANSLATION_CANCELLED etc.), which its own teardown would re-apply after
    # this one and leak into later tests' subprocesses; then the environment from before the test.
    monkeypatch.undo()
    os.chdir(cwd)
    os.environ.clear()
    os.environ.update(original)


def _run(fn, *args, **kwargs):
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as exc:  # the frozen code may raise; the new code must raise the same
        return ("raise", type(exc).__name__, str(exc))


# =============================================================================================
# H: hygiene
# =============================================================================================

def test_core_is_gui_free_cheap_and_python_310():
    source = (SRC / "parallel_epub_core.py").read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 10))
    code = (
        "import sys; sys.modules['PySide6'] = None; sys.path.insert(0, %r); "
        "import parallel_epub_core as p; "
        "heavy = [m for m in ('translator_gui', 'dpi_setup', 'parallel_epub_glossary', 'PySide6', "
        "'extract_glossary_from_epub', 'unified_api_client') if sys.modules.get(m)]; "
        "print(heavy, p.parallel_epub_working_filename('a/b.EPUB'), len(p.__all__))" % str(SRC)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == f"[] b.EPUB {len(pec.__all__)}"
    assert all(hasattr(pec, name) for name in pec.__all__)


@pytest.mark.parametrize("name", ["parallel_epub_core.py", "parallel_epub_glossary.py", "translator_gui.py"])
def test_line_endings_are_uniform_and_bom_unchanged(name):
    data = (SRC / name).read_bytes()
    assert data.count(b"\r\n") in (0, data.count(b"\n")), "mixed line endings"
    assert data.startswith(b"\xef\xbb\xbf") == (name == "translator_gui.py")


# =============================================================================================
# V: verbatim
# =============================================================================================

def test_moved_block_is_verbatim():
    block = "\n".join(git_text(PEG).split("\n")[51:512])
    assert block.startswith("DEFAULT_PARALLEL_EPUB_PROFILE = ") and block.endswith("    return output_path")
    assert block in current_text("parallel_epub_core.py")


@pytest.mark.parametrize("name", sorted(MOVED_BODIES))
def test_lifted_bodies_are_verbatim(name):
    expected = frozen_body(name)
    actual = function_body(current_text("parallel_epub_core.py"), name)
    assert actual == expected


def _changed_spans(legacy, current):
    import difflib
    spans = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, legacy, current, autojunk=False).get_opcodes():
        if tag != "equal":
            spans.append((i1 + 1, i2, j2 - j1))
    return spans


#: parallel_epub_glossary.py rewire spans (frozen first line, frozen last line, new line count).
PEG_REWIRE_SPANS = (
    # docstring note; imports only the moved code used (html, re, uuid, ebooklib, special_file_flags)
    (7, 6, 4), (11, 11, 0), (13, 13, 0), (16, 16, 0), (20, 21, 0),
    # lines 52-512 -> the parallel_epub_core re-export block
    (52, 512, 37),
    # __init__ profiles, _default_chapter_loader, _start_epub_load's loader
    (795, 799, 1), (802, 807, 1), (814, 818, 1), (1145, 1165, 3),
    # restore_persisted_selection, _apply_pending_persisted_mapping
    (1280, 1280, 2), (1282, 1292, 2), (1294, 1303, 1), (1450, 1460, 3), (1468, 1467, 1), (1472, 1484, 1),
    (1493, 1493, 1),
    # _translated_mapping_label, _apply_mapping_offset
    (1536, 1538, 1), (1546, 1547, 3), (1549, 1549, 1), (1555, 1557, 0), (1567, 1567, 1), (1571, 1585, 0),
    (1592, 1597, 1),
    # _set_rows_unmapped, _selected_mapping, _unpaired_file_counts, _unpaired_warning_text
    (1645, 1651, 1), (1674, 1674, 1), (1677, 1677, 1), (1682, 1688, 1), (1693, 1696, 3), (1701, 1711, 2),
    # _update_mapping_status, _persist_prompt_settings, _accept_pair
    (1747, 1768, 7), (1770, 1770, 0), (1774, 1774, 0), (1840, 1845, 6), (1855, 1873, 0), (1875, 1881, 0),
    (1883, 1885, 0), (1887, 1896, 16), (1904, 1917, 3),
)


def test_dialog_module_changed_only_in_rewired_spans():
    legacy = git_text(PEG).split("\n")
    current = current_text("parallel_epub_glossary.py").split("\n")
    assert tuple(_changed_spans(legacy, current)) == PEG_REWIRE_SPANS


@needs_qt
def test_reexports_are_the_core_objects(new_peg):
    for name in REEXPORTS:
        assert getattr(new_peg, name) is getattr(pec, name), name
    assert pec.chapter_special_flags is pec._chapter_special_flags
    assert pec.render_parallel_epub_wrapper is pec.apply_parallel_epub_wrapper


# =============================================================================================
# random chapter sets
# =============================================================================================

_STEMS = ("chapter", "Chapter", "Section", "part", "Text/chapter", "OEBPS/Text/ch", "story", "")


def _rand_filename(rng, n):
    kind = rng.random()
    if kind < 0.12:
        return rng.choice(["cover.xhtml", "nav.xhtml", "0000_Information.xhtml", "Text/title.xhtml",
                           "afterword.xhtml", "Text/Message.xhtml", "notice.xhtml", "toc.xhtml", "author.xhtml"])
    if kind < 0.20:
        return f"{n:04d}_Chapter_{n}_{rng.choice(['Title', 'Notice', 'Arrival', 'Message', 'Author'])}.xhtml"
    stem = rng.choice(_STEMS)
    width = rng.choice((0, 2, 3, 4))
    num = f"{n:0{width}d}" if width else str(n)
    sep = rng.choice(("", "_", "-", " "))
    ext = rng.choice((".xhtml", ".html", ".htm", ".XHTML"))
    return f"{stem}{sep}{num}{ext}"


def _rand_chapters(rng, *, offset=0, max_n=12):
    count = rng.randint(0, max_n)
    names = []
    for i in range(count):
        name = _rand_filename(rng, i + offset)
        if rng.random() < 0.05 and names:
            name = rng.choice(names)  # duplicate names happen in broken EPUBs
        names.append(name)
    return [{"filename": name, "text": f"text {i} of {name}" if rng.random() > 0.05 else ""}
            for i, name in enumerate(names)]


def _rand_shape(rng, chapter):
    kind = rng.random()
    if kind < 0.4:
        return chapter
    if kind < 0.7:
        return (chapter["text"], chapter["filename"])
    if kind < 0.8:
        return [chapter["text"], chapter["filename"], {"extra": 1}]
    if kind < 0.9:
        return chapter["text"]
    return rng.choice([None, (), [], {"filename": None, "text": None}, 5])


_SPECIAL_WORDS = ("message", "title", "author", "notice", "cover", "nav", "toc", "information")


def _rand_predicate(rng):
    if rng.random() < 0.25:
        return None
    words = rng.sample(_SPECIAL_WORDS, rng.randint(1, 4))
    return lambda filename, _w=tuple(words): any(w in str(filename).casefold() for w in _w)


def _rand_reading_order(rng, chapters):
    if rng.random() < 0.4:
        return None
    order = [c["filename"] for c in chapters]
    for _ in range(rng.randint(0, 3)):
        order.insert(rng.randint(0, len(order)), rng.choice(["image_only.xhtml", "Text/illustration.xhtml"]))
    return order


# =============================================================================================
# D: the moved functions vs the frozen module
# =============================================================================================

@needs_qt
def test_moved_functions_match_frozen(frozen_peg):
    rng = random.Random(SEED)
    for state in range(STATES):
        raw = _rand_chapters(rng)
        translated = _rand_chapters(rng, offset=rng.randint(-2, 2))
        kwargs = dict(enable_auto_offset=rng.random() < 0.6, special_file_predicate=_rand_predicate(rng),
                      protect_interior_special_files=rng.random() < 0.6,
                      raw_reading_order=_rand_reading_order(rng, raw),
                      translated_reading_order=_rand_reading_order(rng, translated))
        assert _run(pec.auto_map_epub_chapters, raw, translated, **kwargs) == \
            _run(frozen_peg.auto_map_epub_chapters, raw, translated, **kwargs), state
        shaped = [_rand_shape(rng, c) for c in raw]
        for item in shaped:
            assert (pec.chapter_filename(item), pec.chapter_text(item)) == \
                (frozen_peg.chapter_filename(item), frozen_peg.chapter_text(item)), state
        mapping = []
        for entry in pec.auto_map_epub_chapters(raw, translated):
            if entry["translated_index"] is not None and raw and translated:
                mapping.append({"raw_index": entry["raw_index"] + rng.choice((0, 0, 0, 1)),
                                "translated_index": entry["translated_index"],
                                "raw_filename": raw[entry["raw_index"]]["filename"],
                                "translated_filename": translated[entry["translated_index"]]["filename"].upper()
                                if rng.random() < 0.1 else translated[entry["translated_index"]]["filename"]})
        if rng.random() < 0.3:
            mapping.append(rng.choice([None, "junk", {}, {"raw_index": "x", "raw_filename": "nope.xhtml"}]))
        rng.shuffle(mapping)
        assert pec.restore_parallel_epub_pairs(raw, translated, mapping) == \
            frozen_peg.restore_parallel_epub_pairs(raw, translated, mapping), state
        result = {"raw_path": rng.choice(["", "raw.epub", "/x/raw.epub"]), "translated_path": "tr.epub",
                  "pairs": [p if rng.random() > 0.1 else "junk" for p in
                            pec.restore_parallel_epub_pairs(raw, translated, mapping)[0]],
                  "wrapper_prompt": rng.choice(["", "{raw_text}{translated_text}"]),
                  "system_prompt": rng.choice(["", "sys"]), "profile_name": rng.choice(["", "P"])}
        assert pec.compact_parallel_epub_selection(result) == frozen_peg.compact_parallel_epub_selection(result)
        template = rng.choice([pec.DEFAULT_PARALLEL_EPUB_WRAPPER_PROMPT, "{raw_text}|{x}|{translated_text}{{}}",
                               "{raw_filename}", ""])
        texts = {k: rng.choice(["", "a{raw_text}b", "글", "{translated_filename}"]) for k in
                 ("raw_text", "translated_text", "raw_filename", "translated_filename")}
        assert pec.apply_parallel_epub_wrapper(template, **texts) == \
            frozen_peg.apply_parallel_epub_wrapper(template, **texts)
        path = rng.choice(["", "a.epub", "dir/b.EPUB", "c.txt", "noext", "d.epub.bak"])
        assert pec.parallel_epub_working_filename(path) == frozen_peg.parallel_epub_working_filename(path)
    assert pec.default_parallel_epub_system_prompt() == frozen_peg.default_parallel_epub_system_prompt()


# =============================================================================================
# D: dialog methods (frozen class vs working-tree class) on recording fakes
# =============================================================================================

class FakeItem:
    def __init__(self, text="", data=None):
        self._text = text
        self._data = {}
        if data is not None:
            from PySide6.QtCore import Qt
            self._data[Qt.UserRole] = data

    def setData(self, role, value):
        self._data[role] = value

    def data(self, role):
        return self._data.get(role)

    def setText(self, text):
        self._text = text

    def text(self):
        return self._text


class FakeHeader:
    def __init__(self, log):
        self.log = log

    def setSectionResizeMode(self, column, mode):
        self.log.append(("resize", column, int(getattr(mode, "value", mode))))


class FakeTable:
    def __init__(self, rows, log):
        self.rows = rows
        self.log = log
        self.header = FakeHeader(log)

    def rowCount(self):
        return len(self.rows)

    def item(self, row, column):
        if 0 <= row < len(self.rows):
            return self.rows[row].get(column)
        return None

    def setUpdatesEnabled(self, value):
        self.log.append(("updates", bool(value)))

    def horizontalHeader(self):
        return self.header

    def viewport(self):
        return types.SimpleNamespace(update=lambda: self.log.append(("viewport",)))

    def snapshot(self):
        from PySide6.QtCore import Qt
        out = []
        for row in self.rows:
            cells = []
            for col in (1, 2):
                item = row.get(col)
                cells.append(None if item is None else (item.text(), item.data(Qt.UserRole)))
            out.append(tuple(cells))
        return out


class FakeLabel:
    def __init__(self, log):
        self.log = log

    def setText(self, text):
        self.log.append(("label.text", text))

    def setStyleSheet(self, style):
        self.log.append(("label.style", style))

    def setToolTip(self, tip):
        self.log.append(("label.tip", tip))


class FakeEdit:
    def __init__(self, text):
        self.text = text

    def toPlainText(self):
        return self.text

    def setPlainText(self, text):
        self.text = text


class FakeCombo:
    def __init__(self, items, current):
        self.items = list(items)
        self.current = current

    def currentText(self):
        return self.current

    def setCurrentText(self, text):
        self.current = text

    def findText(self, text):
        return self.items.index(text) if text in self.items else -1


class FakeBox:
    def __init__(self, text, answer, log):
        self.text = text
        self.answer = answer
        log.append(("question", text))

    def exec(self):
        return self.answer


class BoxRecorder:
    """Stand-in for the module's QMessageBox (static information / warning; Yes / No)."""

    def __init__(self, log):
        from PySide6.QtWidgets import QMessageBox
        self.Yes = QMessageBox.Yes
        self.No = QMessageBox.No
        self.log = log

    def information(self, _parent, title, text):
        self.log.append(("information", title, text))

    def warning(self, _parent, title, text):
        self.log.append(("warning", title, text))


DIALOG_METHODS = (
    "_translated_mapping_label", "_apply_mapping_offset", "_set_rows_unmapped", "_selected_mapping",
    "_unpaired_file_counts", "_unpaired_warning_text", "_update_mapping_status", "_persist_prompt_settings",
    "_accept_pair", "_apply_pending_persisted_mapping", "restore_persisted_selection",
)


def _fake_dialog(cls, state, log, answer):
    st = copy.deepcopy(state)
    fake = types.SimpleNamespace()
    fake.config = st["config"]
    fake.profiles = st["profiles"]
    fake.raw_path = st["raw_path"]
    fake.translated_path = st["translated_path"]
    fake.raw_chapters = st["raw_chapters"]
    fake.translated_chapters = st["translated_chapters"]
    fake.translated_reading_order = st["translated_reading_order"]
    fake.special_file_predicate = state["predicate"]
    fake._auto_mapping = st["auto_mapping"]
    fake._mapping_offset = st["offset"]
    fake._active_load = st["active_load"]
    fake._pending_loads = st["pending_loads"]
    fake._pending_persisted_selection = st["pending"]
    fake.result_data = None
    fake.mapping_table = FakeTable([{1: FakeItem(text, data), 2: FakeItem(strategy)} if text is not None else {}
                                    for text, data, strategy in st["table"]], log)
    fake.mapping_status = FakeLabel(log)
    fake.wrapper_edit = FakeEdit(st["wrapper"])
    fake.system_prompt_edit = FakeEdit(st["system_prompt"])
    fake.profile_combo = FakeCombo(list(st["profiles"]) + ["Extra"], st["combo"])
    fake.parent = lambda: None
    fake.accept = lambda: log.append(("accept",))
    fake._load_epub = lambda side, path: log.append(("load", side, os.path.basename(path)))
    fake._rebuild_mapping = lambda: log.append(("rebuild",))
    for name in DIALOG_METHODS:
        setattr(fake, name, types.MethodType(getattr(cls, name), fake))
    fake._create_unpaired_warning_box = lambda mapping: FakeBox(fake._unpaired_warning_text(mapping), answer, log)
    return fake


def _dialog_snapshot(fake, log, result):
    return {
        "result": result,
        "table": fake.mapping_table.snapshot(),
        "offset": fake._mapping_offset,
        "pending": fake._pending_persisted_selection,
        "config": fake.config,
        "profiles": fake.profiles,
        "result_data": fake.result_data,
        "wrapper": fake.wrapper_edit.text,
        "system_prompt": fake.system_prompt_edit.text,
        "combo": fake.profile_combo.current,
        "log": list(log),
    }


def _rand_dialog_state(rng, tmp_path):
    raw = [c for c in _rand_chapters(rng) if c["text"]]
    translated = [c for c in _rand_chapters(rng, offset=rng.randint(-2, 2)) if c["text"]]
    predicate = _rand_predicate(rng)
    order = _rand_reading_order(rng, translated)
    auto = pec.auto_map_epub_chapters(raw, translated, enable_auto_offset=rng.random() < 0.6,
                                      special_file_predicate=predicate)
    table = []
    for index, entry in enumerate(auto):
        if rng.random() < 0.04:
            table.append((None, None, None))  # a row whose items are not built yet
            continue
        value = -1 if entry["translated_index"] is None else entry["translated_index"]
        roll = rng.random()
        if roll < 0.15 and translated:
            value = rng.randrange(-1, len(translated))  # manual pick (duplicates possible)
        elif roll < 0.18:
            value = rng.choice([None, "x", "3", 99, -5])
        table.append(("label", value, entry["strategy"]))
    paths = [str(tmp_path / f"book{i}.epub") for i in range(3)] + [str(tmp_path / "missing.epub"),
                                                                  str(tmp_path / "notes.txt")]
    for p in paths[:3] + [paths[4]]:
        Path(p).write_text("x", encoding="utf-8")
    raw_path = rng.choice(paths[:3])
    translated_path = rng.choice(paths[:3] + [raw_path])
    saved = None
    if rng.random() < 0.7:
        pairs = [{"raw_index": i, "translated_index": e["translated_index"], "raw_filename": raw[i]["filename"],
                  "translated_filename": translated[e["translated_index"]]["filename"]}
                 for i, e in enumerate(auto) if e["translated_index"] is not None]
        if rng.random() < 0.3 and pairs:
            pairs.pop(rng.randrange(len(pairs)))
        if rng.random() < 0.2:
            pairs.append(rng.choice(["junk", {"raw_filename": "gone.xhtml", "translated_filename": "x"}]))
        saved = {"raw_path": rng.choice([raw_path, raw_path.upper(), paths[3], paths[4], ""]),
                 "translated_path": rng.choice([translated_path, paths[3], ""]),
                 "mapping": pairs if rng.random() > 0.1 else [],
                 "wrapper_prompt": rng.choice(["", "{raw_text}\n{translated_text}"]),
                 "system_prompt": rng.choice(["", "saved sys"]),
                 "profile_name": rng.choice(["", "Extra", "Missing", pec.DEFAULT_PARALLEL_EPUB_PROFILE])}
    profiles = {pec.DEFAULT_PARALLEL_EPUB_PROFILE: "default", "Extra": "extra prompt"}
    return {
        "config": {"never_consider_in_between_files_as_special": rng.choice([True, False, None, 0]),
                   "parallel_epub_glossary_wrapper_prompt": "w"} if rng.random() < 0.8 else {},
        "profiles": profiles,
        "raw_path": raw_path,
        "translated_path": translated_path,
        "raw_chapters": raw,
        "translated_chapters": translated,
        "translated_reading_order": order,
        "predicate": predicate,
        "auto_mapping": auto,
        "offset": rng.randint(-3, 3),
        "active_load": rng.choice([None, None, None, ("raw", raw_path, 1)]),
        "pending_loads": rng.choice([[], [], [], [("translated", translated_path, 2)]]),
        "pending": copy.deepcopy(saved) if rng.random() < 0.6 else rng.choice([None, "junk"]),
        "saved": saved,
        "table": table,
        "wrapper": rng.choice([pec.DEFAULT_PARALLEL_EPUB_WRAPPER_PROMPT, "{raw_text} only", "",
                               "{translated_text}{raw_text}"]),
        "system_prompt": rng.choice(["  ", "", "System prompt", " padded \n"]),
        "combo": rng.choice([pec.DEFAULT_PARALLEL_EPUB_PROFILE, "Extra", "  ", "New One "]),
    }


def _dialog_ops(rng, state):
    ops = []
    for _ in range(rng.randint(1, 5)):
        kind = rng.choice(("offset", "offset", "unmap", "selected", "status", "unpaired", "pending", "accept",
                           "accept", "persist", "label", "restore"))
        if kind == "offset":
            ops.append(("_apply_mapping_offset", (rng.choice([-2, -1, 1, 1, 2, 0]),)))
        elif kind == "unmap":
            n = len(state["table"])
            ops.append(("_set_rows_unmapped", ([rng.randint(-2, n + 1) for _ in range(rng.randint(0, 4))],)))
        elif kind == "selected":
            ops.append(("_selected_mapping", ()))
        elif kind == "status":
            ops.append(("_update_mapping_status", ()))
        elif kind == "unpaired":
            ops.append(("_unpaired_warning_text", "selected"))
        elif kind == "pending":
            ops.append(("_apply_pending_persisted_mapping", ()))
        elif kind == "accept":
            ops.append(("_accept_pair", ()))
        elif kind == "persist":
            ops.append(("_persist_prompt_settings", ()))
        elif kind == "label":
            ops.append(("_translated_mapping_label", (rng.randint(-2, len(state["translated_chapters"]) + 1),)))
        else:
            ops.append(("restore_persisted_selection", (copy.deepcopy(state["saved"]) if rng.random() < 0.8
                                                       else rng.choice([None, {}, {"mapping": []}]),)))
    return ops


@needs_qt
def test_dialog_methods_match_frozen(frozen_peg, new_peg, tmp_path, monkeypatch):
    rng = random.Random(SEED + 1)
    exercised = {}
    for state_no in range(STATES):
        state = _rand_dialog_state(rng, tmp_path)
        ops = _dialog_ops(rng, state)
        answer = rng.choice(["yes", "no"])
        sides = []
        for module in (frozen_peg, new_peg):
            log = []
            from PySide6.QtWidgets import QMessageBox
            monkeypatch.setattr(module, "QMessageBox", BoxRecorder(log))
            fake = _fake_dialog(module.ParallelEpubPairDialog, state, log,
                                QMessageBox.Yes if answer == "yes" else QMessageBox.No)
            results = []
            for name, args in ops:
                if args == "selected":
                    args = (fake._selected_mapping(),)
                results.append((name, _run(getattr(fake, name), *args)))
                exercised[name] = exercised.get(name, 0) + 1
            sides.append(_dialog_snapshot(fake, log, results))
        assert sides[1] == sides[0], (state_no, ops)
    assert all(exercised.get(name, 0) >= 20 for name in (
        "_apply_mapping_offset", "_set_rows_unmapped", "_accept_pair", "_apply_pending_persisted_mapping",
        "restore_persisted_selection", "_persist_prompt_settings")), exercised


@needs_qt
def test_background_loader_matches_frozen_closure(frozen_peg, new_peg):
    """``_start_epub_load``'s load_in_background: frozen closure vs the working-tree one."""
    rng = random.Random(SEED + 2)
    frozen_src = git_text(PEG)
    new_src = current_text("parallel_epub_glossary.py")

    def closure(module, text):
        tree = ast.parse(text)
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load_in_background")
        source = textwrap.dedent("\n".join(text.split("\n")[node.lineno - 1:node.end_lineno]))
        return compile(source, "<load_in_background>", "exec"), module

    codes = [closure(frozen_peg, frozen_src), closure(new_peg, new_src)]
    for state in range(STATES):
        chapters = [_rand_shape(rng, c) for c in _rand_chapters(rng)]
        failure = rng.random() < 0.1
        outs = []
        for code, module in codes:
            emitted = []

            def loader(_path, _chapters=chapters, _failure=failure):
                if _failure:
                    raise OSError("broken zip")
                return list(_chapters)

            fake = types.SimpleNamespace(chapter_loader=loader,
                                         epubLoadFinished=types.SimpleNamespace(emit=lambda *a: emitted.append(a)))
            ns = dict(vars(module))
            ns.update(self=fake, side="raw", path="p.epub", serial=state)
            exec(code, ns)
            ns["load_in_background"]()
            outs.append(emitted)
        assert outs[1] == outs[0], state


@needs_qt
def test_dialog_profile_setup_matches_frozen_init_lines(frozen_peg):
    """``__init__``'s profile set-up (frozen statements) vs parallel_epub_profiles / active_parallel_epub_profile."""
    text = git_text(PEG).split("\n")
    first = textwrap.dedent("\n".join(text[794:799]))
    second = textwrap.dedent("\n".join(text[801:807]))
    rng = random.Random(SEED + 3)
    for state in range(STATES):
        config = {}
        if rng.random() < 0.8:
            config["parallel_epub_glossary_profiles"] = rng.choice([
                {}, {"A": "a"}, {pec.DEFAULT_PARALLEL_EPUB_PROFILE: "custom"}, "junk", None, ["x"]])
        if rng.random() < 0.7:
            config["parallel_epub_glossary_active_profile"] = rng.choice(["A", "", None, "Missing",
                                                                          pec.DEFAULT_PARALLEL_EPUB_PROFILE])
        fake = types.SimpleNamespace(config=copy.deepcopy(config))
        ns = dict(vars(frozen_peg))
        ns["self"] = fake
        exec(first, ns)
        exec(second, ns)
        profiles = pec.parallel_epub_profiles(copy.deepcopy(config))
        assert profiles == fake.profiles, state
        assert pec.active_parallel_epub_profile(config, profiles) == ns["active"], state


def test_prompt_settings_match_the_frozen_persist_statements():
    rng = random.Random(SEED + 4)
    for _ in range(STATES):
        profiles = {"A": "a", "B": rng.choice(["", "b"])}
        combo = rng.choice(["", "  ", "A", " B ", None]) or ""
        wrapper = rng.choice(["", "{raw_text}", "w"])
        config = {"parallel_epub_glossary_profiles": "old", "x": 1}
        expected = dict(config)
        expected["parallel_epub_glossary_profiles"] = dict(profiles)
        expected["parallel_epub_glossary_active_profile"] = (combo.strip() or pec.DEFAULT_PARALLEL_EPUB_PROFILE)
        expected["parallel_epub_glossary_wrapper_prompt"] = wrapper
        config.update(pec.parallel_epub_prompt_settings(profiles, combo, wrapper))
        assert list(config.items()) == list(expected.items())


# =============================================================================================
# D: TranslatorGUI pair helpers (frozen methods vs the wrappers)
# =============================================================================================

class _PairOwner:
    """Attribute bag standing in for TranslatorGUI in the pair-helper comparisons."""

    def __init__(self, config, log):
        self.config = config
        self._log = log
        self.selected_files = []
        self._parallel_epub_pair_source = None

    def append_log(self, message):
        self._log.append(("log", message))


def _bind(owner, fn, name):
    setattr(owner, name, types.MethodType(fn, owner))


def _fresh_dir(root):
    import shutil
    os.chdir(str(TESTS))
    if Path(root).exists():
        shutil.rmtree(root)
    Path(root).mkdir(parents=True)


def _tree_bytes(root):
    out = {}
    root = Path(root)
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file():
                out[path.relative_to(root).as_posix()] = path.read_bytes()
            else:
                out[path.relative_to(root).as_posix() + "/"] = b""
    return out


@needs_qt
def test_glossary_folder_and_sidecar_helpers_match_frozen(tmp_path, monkeypatch, frozen_tg, tg_module):
    TranslatorGUI = tg_module.TranslatorGUI
    names = ("_resolve_parallel_epub_glossary_output_dir", "_parallel_epub_mapping_sidecar_path",
             "_write_parallel_epub_mapping_sidecar", "_read_parallel_epub_mapping_sidecar")
    raw_names = ["Raw Novel.epub", "raw:novel?.epub", "[123] 소설.epub", "  spaced  .epub", "a.EPUB", "noext"]
    for state in range(STATES):
        sides = []
        root = tmp_path / f"s{state}"
        for side in ("frozen", "new"):
            # both sides run in the same (fresh) folder, so paths need no normalising
            _fresh_dir(root)
            os.chdir(root)
            rng_side = random.Random(f"{SEED}-{state}")
            out_dir = rng_side.choice([str(root / "out"), "", None])
            env_dir = rng_side.choice([None, None, str(root / "env out")])
            if env_dir:
                monkeypatch.setenv("OUTPUT_DIRECTORY", env_dir)
            else:
                monkeypatch.delenv("OUTPUT_DIRECTORY", raising=False)
            log = []
            config = {"output_directory": out_dir} if out_dir is not None else {}
            owner = _PairOwner(config, log)
            raw = str(root / "books" / rng_side.choice(raw_names))
            selected = rng_side.random() < 0.5
            owner._parallel_epub_pair_source = rng_side.choice([
                None, {"raw_path": raw, "generated_path": str(root / "gen.epub")}, {"raw_path": ""}, "junk"])
            owner.selected_files = [str(root / "gen.epub")] if selected else []
            for name in names:
                fn = frozen_tg(name) if side == "frozen" else getattr(TranslatorGUI, name)
                _bind(owner, fn, name)
            _bind(owner, TranslatorGUI._parallel_epub_pair_is_selected, "_parallel_epub_pair_is_selected")
            selection = rng_side.choice([
                {"raw_path": raw, "translated_path": "t.epub", "mapping": [{"raw_index": 0}]},
                {"raw_path": raw, "mapping": []}, {"raw_path": "", "mapping": [{}]}, None, "junk",
                {"raw_path": raw.upper(), "mapping": [{"raw_index": 1, "raw_filename": "Text/a.xhtml"}]},
            ])
            results = [
                _run(owner._resolve_parallel_epub_glossary_output_dir, rng_side.choice(["", raw])),
                _run(owner._resolve_parallel_epub_glossary_output_dir, raw, create=rng_side.random() < 0.5),
                _run(owner._parallel_epub_mapping_sidecar_path, rng_side.choice(["", "  ", raw])),
                _run(owner._write_parallel_epub_mapping_sidecar, copy.deepcopy(selection)),
                _run(owner._read_parallel_epub_mapping_sidecar, rng_side.choice([raw, raw.upper(), "", "x.epub"])),
            ]
            if rng_side.random() < 0.3:
                path = owner._parallel_epub_mapping_sidecar_path(raw)
                if path:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    Path(path).write_text(rng_side.choice(["{bad json", "[]", '{"mapping": {}}',
                                                           json.dumps({"raw_path": raw, "mapping": []})]),
                                          encoding="utf-8")
                results.append(_run(owner._read_parallel_epub_mapping_sidecar, raw))
            sides.append((results, _tree_bytes(root), log))
        assert sides[1] == sides[0], state


def _write_fixture_epub(path, names, *, title="Fixture", text_for=None):
    from ebooklib import epub
    book = epub.EpubBook()
    book.set_identifier(Path(path).stem)
    book.set_title(title)
    book.set_language("en")
    documents = []
    for index, name in enumerate(names):
        document = epub.EpubHtml(title=f"Doc {index}", file_name=name)
        body = text_for(index, name) if text_for else f"Readable text of {name} number {index}."
        document.content = f"<html><body><h1>Doc {index}</h1><p>{body}</p></body></html>"
        book.add_item(document)
        documents.append(document)
    book.spine = documents
    book.toc = tuple(documents)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)
    return str(path)


_STAMP = re.compile(rb"glossarion-parallel-[0-9a-f]{32}|\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z?")


def _epub_members(path):
    with zipfile.ZipFile(path) as archive:
        return {name: _STAMP.sub(b"<STAMP>", archive.read(name)) for name in sorted(archive.namelist())}


#: fixture pairs: (raw document names, translated document names, raw text, translated text)
FIXTURE_PAIRS = {
    "numbered_offset": (
        [f"Text/chapter{n:04d}.xhtml" for n in range(1, 7)],
        ["Text/0000_Information.xhtml"] + [f"Text/{n:04d}_Chapter_{n}.xhtml" for n in range(1, 7)],
        "김상현은 문을 열었다 {n}.", "Kim Sang-hyun opened the door {n}.",
    ),
    "front_matter": (
        ["Text/cover.xhtml", "Text/title.xhtml"] + [f"Text/ch{n}.xhtml" for n in range(1, 5)],
        ["Text/cover.xhtml"] + [f"Text/Section{n:03d}.xhtml" for n in range(1, 6)],
        "第{n}章 林凡走进了大殿。", "Chapter {n}: Lin Fan entered the hall.",
    ),
    "same_names": (
        [f"OEBPS/Text/p{n}.xhtml" for n in range(1, 5)],
        [f"OEBPS/Text/p{n}.xhtml" for n in range(1, 5)],
        "彼は剣を抜いた {n}。", "He drew his sword {n}.",
    ),
}


def _fixture_pair(root, key):
    raw_names, translated_names, raw_text, translated_text = FIXTURE_PAIRS[key]
    raw = _write_fixture_epub(root / f"{key} raw.epub", raw_names,
                              text_for=lambda i, _n: raw_text.format(n=i))
    translated = _write_fixture_epub(root / f"{key} translated.epub", translated_names,
                                     text_for=lambda i, _n: translated_text.format(n=i))
    return raw, translated


class _Recorder:
    def __init__(self, log, name):
        self.log = log
        self.name = name

    def setText(self, text):
        self.log.append((self.name, "text", text))

    def setToolTip(self, text):
        self.log.append((self.name, "tip", text))

    def emit(self, *args):
        self.log.append((self.name, "emit", args))


class _SyncThread:
    """threading.Thread stand-in that runs the target on start()."""

    def __init__(self, target=None, name=None, daemon=None, args=(), kwargs=None):
        self.target, self.args, self.kwargs, self.name = target, args, kwargs or {}, name

    def start(self):
        self.target(*self.args, **self.kwargs)

    def is_alive(self):
        return False


def _owner_for_flow(tg_module, frozen_tg, side, root, log, config):
    TranslatorGUI = tg_module.TranslatorGUI
    owner = _PairOwner(config, log)
    owner.entry_epub = _Recorder(log, "entry")
    owner.file_status_label = _Recorder(log, "status")
    owner.save_config = lambda show_message=True: log.append(("save_config", show_message))
    owner.parallel_epub_restore_finished_signal = _Recorder(log, "restore_signal")
    owner.glossary_thread = None
    owner.glossary_future = None
    for name in ("_resolve_parallel_epub_glossary_output_dir", "_parallel_epub_mapping_sidecar_path",
                 "_write_parallel_epub_mapping_sidecar", "_read_parallel_epub_mapping_sidecar",
                 "_load_parallel_epub_chapters", "_build_parallel_epub_pair_artifact",
                 "_activate_parallel_epub_pair_source", "_start_parallel_epub_pair_restore",
                 "_finish_parallel_epub_pair_restore"):
        fn = frozen_tg(name) if side == "frozen" else getattr(TranslatorGUI, name)
        _bind(owner, fn, name)
    for name in ("_release_parallel_epub_pair_source", "_parallel_epub_pair_is_selected"):
        _bind(owner, getattr(TranslatorGUI, name), name)
    return owner


def _normalized_state(owner, root, temp_names):
    """The pair record without its temporary folder (the working EPUB keeps only its name)."""
    state = dict(owner._parallel_epub_pair_source or {})
    temp = state.pop("temporary_directory", None)
    if state.get("generated_path"):
        temp_names.append(os.path.dirname(state["generated_path"]))
        state["generated_path"] = os.path.basename(state["generated_path"])
    state["has_temp_dir"] = temp is not None
    return copy.deepcopy(state)


@needs_qt
@pytest.mark.parametrize("key", sorted(FIXTURE_PAIRS))
def test_fixture_pair_desktop_flow_matches_frozen(key, tmp_path, monkeypatch, frozen_tg, tg_module, frozen_peg):
    """Load both fixture EPUBs, map, accept, activate (sidecar + working EPUB), then restore the
    saved pair on "launch": frozen TranslatorGUI / dialog code vs the working tree."""
    sync = types.SimpleNamespace(Thread=_SyncThread)
    monkeypatch.setattr(tg_module, "threading", sync)
    monkeypatch.setitem(frozen_tg("_start_parallel_epub_pair_restore").__globals__, "threading", sync)
    sides = []
    root = tmp_path / "flow"
    for side in ("frozen", "new"):
        # both sides run in the same (fresh) folder, so paths need no normalising
        _fresh_dir(root)
        os.chdir(root)
        raw, translated = _fixture_pair(root, key)
        log = []
        config = {"output_directory": str(root / "out"), "never_consider_in_between_files_as_special": True}
        owner = _owner_for_flow(tg_module, frozen_tg, side, root, log, config)
        module = frozen_peg if side == "frozen" else sys.modules["parallel_epub_glossary"]
        swap = swapped_module("parallel_epub_glossary", frozen_peg) if side == "frozen" else contextlib.nullcontext()
        temps = []
        with swap:
            raw_chapters = owner._load_parallel_epub_chapters(raw)
            translated_chapters = owner._load_parallel_epub_chapters(translated)
            # the dialog's background loader + auto map + accept (pure dialog statements)
            loaded = []
            for chapters in (raw_chapters, translated_chapters):
                loaded.append(pec.load_parallel_epub_documents(lambda _p, _c=chapters: _c, "x"))
            (raw_docs, raw_order, _e1), (tr_docs, tr_order, _e2) = loaded
            mapping = [{"raw_index": e["raw_index"], "translated_index": e["translated_index"]}
                       for e in module.auto_map_epub_chapters(raw_docs, tr_docs, raw_reading_order=raw_order,
                                                              translated_reading_order=tr_order)
                       if e["translated_index"] is not None]
            pairs = [{"raw_index": m["raw_index"], "translated_index": m["translated_index"],
                      "raw_filename": raw_docs[m["raw_index"]]["filename"],
                      "raw_text": raw_docs[m["raw_index"]]["text"],
                      "translated_filename": tr_docs[m["translated_index"]]["filename"],
                      "translated_text": tr_docs[m["translated_index"]]["text"]} for m in mapping]
            result = {"raw_path": raw, "translated_path": translated, "pairs": pairs,
                      "wrapper_prompt": module.DEFAULT_PARALLEL_EPUB_WRAPPER_PROMPT,
                      "system_prompt": "Use the established names.", "profile_name": "Parallel EPUB Glossary"}
            owner._activate_parallel_epub_pair_source(result)
            activated = _normalized_state(owner, root, temps)
            working = _epub_members(owner._parallel_epub_pair_source["generated_path"])
            sidecar = Path(owner._parallel_epub_pair_source["mapping_sidecar_path"]).read_bytes()
            # next launch: rebuild from config + sidecar, then activate on the GUI thread
            owner._start_parallel_epub_pair_restore()
            payload = next(entry[2] for entry in log if entry[:2] == ("restore_signal", "emit"))
            restore_error = payload[1]
            owner._finish_parallel_epub_pair_restore(*payload)
            restored = _normalized_state(owner, root, temps)
            restored_epub = _epub_members(owner._parallel_epub_pair_source["generated_path"])
            owner._release_parallel_epub_pair_source()
        text_log = json.dumps(log, default=repr, ensure_ascii=False)
        for temp in temps:
            text_log = text_log.replace(json.dumps(temp)[1:-1], "<TEMP>")
        text_log = re.sub(r"<[^<>]*TemporaryDirectory[^<>]*>", "<TemporaryDirectory>", text_log)
        sides.append({"activated": activated, "working": working, "sidecar": sidecar, "restored": restored,
                      "restored_epub": restored_epub, "config": copy.deepcopy(config), "log": json.loads(text_log),
                      "out": _tree_bytes(root / "out"), "pairs": len(pairs), "restore_error": restore_error})
    assert sides[0]["pairs"] > 0 and not sides[0]["restore_error"]
    assert sides[1] == sides[0]


@needs_qt
@pytest.mark.parametrize("key", sorted(FIXTURE_PAIRS))
def test_fixture_pair_mobile_path_matches_desktop(key, tmp_path, monkeypatch, tg_module, frozen_tg):
    """The GUI-free path a mobile job takes (rebuild from the saved selection, working EPUB,
    pair record) produces the frozen desktop's working EPUB, record and sidecar."""
    root = tmp_path / "desktop"
    root.mkdir()
    os.chdir(root)
    raw, translated = _fixture_pair(root, key)
    log = []
    config = {"output_directory": str(root / "out")}
    owner = _owner_for_flow(tg_module, frozen_tg, "frozen", root, log, config)
    docs = [pec.load_parallel_epub_documents(pec.load_parallel_epub_chapters, p) for p in (raw, translated)]
    (raw_docs, raw_order, _e1), (tr_docs, tr_order, _e2) = docs
    auto = pec.auto_map_epub_chapters(raw_docs, tr_docs, raw_reading_order=raw_order, translated_reading_order=tr_order)
    mapping = pec.selected_parallel_epub_mapping([-1 if e["translated_index"] is None else e["translated_index"]
                                                  for e in auto])
    assert pec.validate_parallel_epub_pair(
        loading=False, raw_path=raw, translated_path=translated, raw_chapters=raw_docs, translated_chapters=tr_docs,
        wrapper_prompt=pec.DEFAULT_PARALLEL_EPUB_WRAPPER_PROMPT, system_prompt="sys", mapping=mapping) is None
    result = {"raw_path": raw, "translated_path": translated,
              "pairs": pec.build_parallel_epub_pairs(mapping, raw_docs, tr_docs),
              "wrapper_prompt": pec.DEFAULT_PARALLEL_EPUB_WRAPPER_PROMPT, "system_prompt": "sys",
              "profile_name": pec.DEFAULT_PARALLEL_EPUB_PROFILE}
    owner._activate_parallel_epub_pair_source(result)
    desktop_state = dict(owner._parallel_epub_pair_source)
    desktop_epub = _epub_members(desktop_state["generated_path"])
    desktop_sidecar = Path(desktop_state["mapping_sidecar_path"]).read_bytes()

    # mobile: read the sidecar the desktop wrote, rebuild, write the working EPUB, build the record
    selection = pec.read_parallel_epub_mapping_sidecar(raw, config)
    assert selection == json.loads(desktop_sidecar.decode("utf-8")) == desktop_state["persistent_selection"]
    rebuilt, skipped = pec.rebuild_parallel_epub_pair_result(selection, config)
    assert skipped == 0
    temp_dir, generated = pec.build_parallel_epub_pair_artifact(rebuilt)
    try:
        assert _epub_members(generated) == desktop_epub
        state = pec.parallel_epub_pair_source_state(
            rebuilt, persistent_selection=pec.compact_parallel_epub_selection(rebuilt),
            mapping_sidecar_path=desktop_state["mapping_sidecar_path"], generated_path=generated,
            pair_temp_dir=temp_dir)
        for name in ("raw_path", "translated_path", "system_prompt", "profile_name", "pair_count",
                     "raw_filenames", "persistent_selection", "mapping_sidecar_path"):
            assert state[name] == desktop_state[name], name
    finally:
        temp_dir.cleanup()
        owner._release_parallel_epub_pair_source()


# =============================================================================================
# S: the real dialog, offscreen (frozen class vs working-tree class)
# =============================================================================================

def _wait_idle(dialog, qapp, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        qapp.processEvents()
        if dialog._active_load is None and not dialog._pending_loads and not dialog._mapping_building:
            if dialog.raw_chapters and dialog.translated_chapters:
                return True
        time.sleep(0.01)
    return False


def _real_dialog_snapshot(dialog):
    from PySide6.QtCore import Qt
    table = dialog.mapping_table
    rows = []
    for row in range(table.rowCount()):
        rows.append(tuple(
            (table.item(row, col).text(), table.item(row, col).data(Qt.UserRole)) if table.item(row, col) else None
            for col in (0, 1, 2)))
    return {"rows": rows, "status": dialog.mapping_status.text(), "style": dialog.mapping_status.styleSheet(),
            "offset": dialog._mapping_offset, "result": dialog.result_data, "config": dialog.config,
            "system_prompt": dialog.system_prompt_edit.toPlainText(), "profile": dialog.profile_combo.currentText()}


@needs_qt
def test_real_dialog_offscreen_matches_frozen(qapp, frozen_peg, new_peg, tmp_path, monkeypatch):
    rng = random.Random(SEED + 6)
    keys = sorted(FIXTURE_PAIRS)
    fixtures = {key: _fixture_pair(tmp_path, key) for key in keys}
    for run in range(DIALOG_RUNS):
        key = keys[run % len(keys)]
        raw, translated = fixtures[key]
        steps = [rng.choice([1, -1, 2, 0]) for _ in range(rng.randint(0, 3))]
        unmap = sorted({rng.randint(0, 4) for _ in range(rng.randint(0, 2))})
        answer_yes = rng.random() < 0.8
        snapshots = []
        for module in (frozen_peg, new_peg):
            from PySide6.QtWidgets import QMessageBox
            boxes = []
            monkeypatch.setattr(module, "QMessageBox", BoxRecorder(boxes))
            config = {"never_consider_in_between_files_as_special": True}
            predicate = (lambda name: any(w in name.casefold() for w in ("cover", "title", "information")))
            with (swapped_module("parallel_epub_glossary", frozen_peg) if module is frozen_peg
                  else contextlib.nullcontext()):
                dialog = module.ParallelEpubPairDialog(config=config, special_file_predicate=predicate)
                dialog._load_epub("raw", raw)
                dialog._load_epub("translated", translated)
                assert _wait_idle(dialog, qapp), "the background loader did not finish"
                states = [_real_dialog_snapshot(dialog)]
                for delta in steps:
                    dialog._apply_mapping_offset(delta)
                    states.append(_real_dialog_snapshot(dialog))
                dialog._set_rows_unmapped(unmap)
                states.append(_real_dialog_snapshot(dialog))
                dialog._create_unpaired_warning_box = lambda mapping, _d=dialog: FakeBox(
                    _d._unpaired_warning_text(mapping), QMessageBox.Yes if answer_yes else QMessageBox.No, boxes)
                dialog._accept_pair()
                states.append(_real_dialog_snapshot(dialog))
                selection = pec.compact_parallel_epub_selection(dialog.result_data) if dialog.result_data else None
                if selection:
                    dialog.clear_selection()
                    dialog.restore_persisted_selection(selection)
                    assert _wait_idle(dialog, qapp)
                    states.append(_real_dialog_snapshot(dialog))
                dialog.deleteLater()
                qapp.processEvents()
            snapshots.append((states, boxes))
        assert snapshots[1] == snapshots[0], (key, steps, unmap)
