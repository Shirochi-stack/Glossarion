"""U8: manga_editor_core (the GUI-free manga editor) parity and API tests.

ImageRenderer's GUI-free editor functions (Detect / Clean / Recognize / Translate / Translate All,
the per-box actions, Save & Update Overlay re-render, per-image state) moved verbatim into
``manga_editor_core``; ImageRenderer re-binds the same code objects to its own namespace
(``bind_editor_namespace``), so on the desktop they keep calling its Qt helpers. Five moved
bodies carry listed edits (Qt lines -> hooks, the google_vision_rest fallback); five
helpers were split out of ImageRenderer's Qt dialogs and two out of manga_image_preview's Stop
button; ``ImageStateManager`` moved out of manga_integration with its spawn worker gated by
``mobile_runtime.processes_available()``. The oracle is the desktop at ``U8_BASE_SHA`` (main
9355bb5d, the parent of the move, which already reads untrusted images through
``safe_image.open_image``), read with ``git show``.

* V: every moved / split body equals the frozen one (modulo the listed edits); the desktop modules
  keep everything else byte-for-byte; the desktop hooks hold the replaced lines;
* B / I: the desktop binding (same code objects, ImageRenderer globals, every free name resolves
  in both namespaces); the module imports without Qt and parses as Python 3.10;
* S: ImageStateManager without its worker process on mobile, unchanged on the desktop, and
  equal to the frozen class on the same operations;
* T: trace parity of the per-image pipeline (frozen ImageRenderer vs the desktop binding vs the
  mobile namespace) with stubbed detector / OCR / inpainter / LLM: update-queue messages, logs,
  page state and written images;
* R: image-diff of the PIL re-render (_render_with_manga_translator, Save & Update Overlay,
  imported-session render, single-overlay update) on a fixture page with a fixed font
  (tests/fixtures/manga_editor/DejaVuSans-Latin.ttf: DejaVu Sans, Latin subset; Bitstream Vera /
  DejaVu licence in LICENSE-DejaVu.txt), frozen vs desktop vs mobile;
* Q: offscreen smoke of the preview editor actions on a real MangaImagePreviewWidget (Detect,
  Recognize, Translate, Edit OCR / Edit Translation dialogs, Set Inpainting Iterations, Mark as
  free text, Clean This Rectangle, Delete, Stop), frozen vs rewired ImageRenderer;
* M: the mobile MangaEditorSession (HeadlessOwner-compatible main_gui, no worker process):
  workflow steps, per-box actions, Save & Update Overlay, stop, OCR JSON export / import;
* G: the editor OCR's google_vision_rest fallback when the SDK is missing.
"""

from __future__ import annotations

import ast
import builtins
import functools
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import manga_editor_core as core  # noqa: E402

#: main at the owner's security upgrade (parent of the U8 move): the frozen desktop oracle.
U8_BASE_SHA = "9355bb5d376ca9eb8149a36b35b6028a95f86009"
FONT_PATH = TESTS_DIR / "fixtures" / "manga_editor" / "DejaVuSans-Latin.ttf"


# ===========================================================================
# Frozen sources
# ===========================================================================

@functools.lru_cache(maxsize=None)
def frozen_source(relpath):
    raw = subprocess.check_output(["git", "show", f"{U8_BASE_SHA}:{relpath}"], cwd=str(REPO_ROOT))
    return raw.decode("utf-8").replace("\r\n", "\n")


def current_source(name):
    return (SRC_DIR / name).read_text(encoding="utf-8").replace("\r\n", "\n")


def _top_defs(source):
    tree = ast.parse(source)
    lines = source.split("\n")
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
            out[node.name] = "\n".join(lines[start:node.end_lineno]) + "\n"
    return out


def _methods(source, class_name):
    tree = ast.parse(source)
    lines = source.split("\n")
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name][0]
    return {n.name: "\n".join(lines[n.lineno - 1:n.end_lineno]) + "\n"
            for n in cls.body if isinstance(n, ast.FunctionDef)}


def _is_comment(line):
    return line.strip().startswith("#")


def _canonical_block(lines):
    """Block lines relative to their first code line's indentation; comment-only lines stripped,
    whitespace-only lines empty (the split helpers re-indent their lifted block)."""
    code = [line for line in lines if line.strip() and not _is_comment(line)]
    base = len(code[0]) - len(code[0].lstrip(" "))
    out = []
    for line in lines:
        if not line.strip():
            out.append("")
        elif _is_comment(line):
            out.append(line.strip())
        else:
            assert line.startswith(" " * base), line
            out.append(line[base:].rstrip())
    return out


def _marker_block(text, start, end, end_after=None):
    lines = text.split("\n")
    s = next(i for i, line in enumerate(lines) if line.strip() == start.strip())
    a = s if end_after is None else next(i for i in range(s, len(lines)) if lines[i].strip() == end_after.strip())
    e = next(i for i in range(a, len(lines)) if lines[i].strip() == end.strip())
    return lines[s:e + 1]


def _function_body(source):
    """Body lines of a single function source (no def line / docstring)."""
    node = ast.parse(source).body[0]
    lines = source.split("\n")
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
        start = first.end_lineno
    else:
        start = first.lineno - 1
    return lines[start:node.end_lineno]


def _replace_stripped(lines, old, new):
    old_lines = [line.strip() for line in old.strip("\n").split("\n")]
    hits = [i for i in range(len(lines) - len(old_lines) + 1)
            if [line.strip() for line in lines[i:i + len(old_lines)]] == old_lines]
    assert len(hits) == 1, (old, hits)
    i = hits[0]
    indent = len(lines[i]) - len(lines[i].lstrip(" "))
    return lines[:i] + [" " * indent + line for line in new.strip("\n").split("\n")] + lines[i + len(old_lines):]


# ===========================================================================
# The move specification (what the oracle comparison allows)
# ===========================================================================

_STYLE_OLD = (
    "from PySide6.QtGui import QPen, QBrush, QColor\n"
    "rect_item.setPen(QPen(QColor(0, 150, 255), 2))  # Blue border\n"
    "rect_item.setBrush(QBrush(QColor(0, 150, 255, 50)))  # Semi-transparent blue fill\n"
)


def _indent(text, n):
    return "".join(" " * n + line if line.strip() else line for line in text.splitlines(True))


#: (function, frozen text, new text): the only differences between a moved body and the frozen one.
EDITS = (
    ("_run_ocr_on_regions",
     "            try:\n                from google.cloud import vision\n                import io\n",
     "            try:\n                vision = _import_google_vision()\n                import io\n"),
    ("_update_rectangles_with_recognition", _indent(_STYLE_OLD, 16), _indent("_style_recognized_rectangle(rect_item)\n", 16)),
    ("_process_ocr_result", _indent(_STYLE_OLD, 12), _indent("_style_recognized_rectangle(rect_item)\n", 12)),
    ("_handle_clean_this_rectangle",
     "        from PySide6.QtWidgets import QMessageBox\n        from PySide6.QtCore import QThread\n", ""),
    ("_handle_clean_this_rectangle",
     "        if is_excluded:\n"
     "            reply = QMessageBox.question(\n"
     "                self.dialog,\n"
     "                \"Rectangle Excluded\",\n"
     "                f\"Rectangle {region_index} is currently excluded from inpainting.\\n\\nDo you want to clean it anyway?\",\n"
     "                QMessageBox.Yes | QMessageBox.No,\n"
     "                QMessageBox.No\n"
     "            )\n"
     "            if reply == QMessageBox.No:\n"
     "                print(f\"[CLEAN_RECT] User cancelled - rectangle {region_index} is excluded\")\n"
     "                return\n",
     "        if is_excluded:\n"
     "            if not _confirm_clean_excluded_rectangle(self, region_index):\n"
     "                return\n"),
    ("_render_with_manga_translator",
     "        if refresh_preview:\n"
     "            from PySide6.QtCore import QTimer\n"
     "            QTimer.singleShot(0, lambda: _load_rendered_image_to_output_tab(self, rendered_pil, output_path, switch_tab))\n",
     "        if refresh_preview:\n"
     "            _schedule_rendered_output_refresh(self, rendered_pil, output_path, switch_tab)\n"),
)

#: Split helpers: (helper, source function, start marker, end marker, in-block substitutions,
#: prologue / epilogue lines added around the lifted block, the call left in the source).
SPLITS = (
    ("_manual_translate_prompt", "_add_context_menu_to_rectangle",
     "# Get manual edit settings from config",
     "actual_prompt = translate_prompt.replace('{language}', target_language)",
     (), [], ["return actual_prompt, target_language"],
     "actual_prompt, target_language = _manual_translate_prompt(self)"),
    ("_apply_ocr_text_edit", "_show_ocr_popup",
     "if new_text != ocr_text and region_index is not None:",
     "print(f\"[DEBUG] Updated OCR text for region {region_index}: '{new_text[:50]}...'\")",
     (), [], [],
     "_apply_ocr_text_edit(self, region_index, ocr_text, new_text)"),
    ("_apply_translation_text_edit", "_show_translation_popup",
     "# Update the stored data",
     "print(f\"[DEBUG] Updated translation for region {region_index}\")",
     (), ["changed = False"], ["return changed"],
     "changed = _apply_translation_text_edit(self, region_index, original, translation, new_original, new_translation)"),
    ("_persist_translation_text_edit", "_show_translation_popup",
     "# Persist updated translated_texts to state so overlays restore across sessions",
     "self._log(f\"⚠️ Failed to persist translation change: {persist_err}\", \"warning\")",
     (("dialog.accept()\nreturn\n", "return False\n"),), [], ["return True"],
     "if not _persist_translation_text_edit(self, region_index, new_original, new_translation):\n"
     "    dialog.accept()\n"
     "    return"),
    ("_apply_inpaint_iterations", "_handle_set_inpainting_iterations",
     "if value == -1:",
     "print(f\"[INPAINT_ITERATIONS] Error updating state: {e}\")",
     (), [], [],
     "_apply_inpaint_iterations(self, region_index, rect_item, value)"),
)

#: manga_image_preview Stop button: (helper, start, end_after, end, call left in the button).
PREVIEW_SPLITS = (
    ("_request_force_stop", "if mi and hasattr(mi, '_log'):", "import TransateKRtoEN", "pass",
     "# Flags, env and module-level hard cancel, shared with the mobile editor (U8)\n"
     "import manga_editor_core\n"
     "manga_editor_core._request_force_stop(mi)"),
    ("_request_graceful_stop", "if hasattr(mi, '_log'):", "if hasattr(mi, '_global_cancellation'):",
     "print(\"[STOP] Set _global_cancellation on manga_integration\")",
     "# Editor-level stop flags, shared with the mobile editor (U8)\n"
     "import manga_editor_core\n"
     "manga_editor_core._request_graceful_stop(mi)"),
)

GATE_LINES = (
    "            # Mobile (U8): no worker process on Android/iOS. The state stays in this process and\n"
    "            # flush() / flush_async() write it (the worker only mirrored it).\n"
    "            if not mobile_runtime.processes_available():\n"
    "                self._mp_enabled = False\n"
    "                return\n"
    "            \n"
)

DESKTOP_HOOK_BODIES = {
    "_style_recognized_rectangle": _STYLE_OLD,
    "_schedule_rendered_output_refresh": (
        "from PySide6.QtCore import QTimer\n"
        "QTimer.singleShot(0, lambda: _load_rendered_image_to_output_tab(self, rendered_pil, output_path, switch_tab))\n"
    ),
}

MOVED = tuple(name for name in core.EDITOR_FUNCTIONS if name not in {s[0] for s in SPLITS})


# ===========================================================================
# V: verbatim
# ===========================================================================

def test_registry_is_consistent():
    assert set(core.SPLIT_HELPERS) == {s[0] for s in SPLITS} | {s[0] for s in PREVIEW_SPLITS}
    assert set(core.EDITED_FUNCTIONS) == {e[0] for e in EDITS}
    assert set(core.EDITOR_FUNCTIONS) == set(MOVED) | {s[0] for s in SPLITS}
    frozen = _top_defs(frozen_source("src/ImageRenderer.py"))
    for name in MOVED:
        assert name in frozen, name


@pytest.mark.parametrize("name", MOVED)
def test_moved_function_is_verbatim(name):
    frozen = _top_defs(frozen_source("src/ImageRenderer.py"))[name]
    for func, old, new in EDITS:
        if func == name:
            assert frozen.count(old) == 1, (name, old)
            frozen = frozen.replace(old, new)
    assert _top_defs(current_source("manga_editor_core.py"))[name] == frozen


@pytest.mark.parametrize("split", SPLITS, ids=[s[0] for s in SPLITS])
def test_split_helper_is_lifted_verbatim(split):
    helper, source_name, start, end, subs, prologue, epilogue, _call = split
    frozen_fn = _top_defs(frozen_source("src/ImageRenderer.py"))[source_name]
    block = _canonical_block(_marker_block(frozen_fn, start, end))
    for old, new in subs:
        block = _replace_stripped(block, old, new)
    expected = prologue + block + epilogue
    body = _canonical_block(_function_body(_top_defs(current_source("manga_editor_core.py"))[helper]))
    assert body == expected


@pytest.mark.parametrize("split", PREVIEW_SPLITS, ids=[s[0] for s in PREVIEW_SPLITS])
def test_preview_stop_helper_is_lifted_verbatim(split):
    helper, start, end_after, end, _call = split
    frozen_fn = _methods(frozen_source("src/manga_image_preview.py"), "MangaImagePreviewWidget")["_on_stop_translation_clicked"]
    block = _canonical_block(_marker_block(frozen_fn, start, end, end_after))
    body = _canonical_block(_function_body(_top_defs(current_source("manga_editor_core.py"))[helper]))
    assert body == block


def _frozen_state_block():
    source = frozen_source("src/manga_integration.py")
    start = source.index("# Module-level worker function for state management (must be picklable)\n")
    end = source.index("class _MangaGuiLogHandler(logging.Handler):")
    return source[start:end].rstrip("\n") + "\n"


def test_image_state_manager_is_verbatim_except_the_mobile_gate():
    current = current_source("manga_editor_core.py")
    start = current.index("# Module-level worker function for state management (must be picklable)\n")
    end = current.index("\n\n\n", current.index("    def __del__(self):"))
    moved = current[start:end].rstrip("\n") + "\n"
    assert moved.count(GATE_LINES) == 1
    assert moved.replace(GATE_LINES, "") == _frozen_state_block()


def test_desktop_image_renderer_keeps_everything_else():
    frozen = _top_defs(frozen_source("src/ImageRenderer.py"))
    current = _top_defs(current_source("ImageRenderer.py"))
    for name in MOVED:
        assert name not in current, f"ImageRenderer still defines {name}"
    split_by_source = {}
    for helper, source_name, start, end, _subs, _pro, _epi, call in SPLITS:
        split_by_source.setdefault(source_name, []).append((start, end, call))
    for name, text in frozen.items():
        if name in MOVED:
            continue
        if name in split_by_source:
            lines = text.split("\n")
            for start, end, call in split_by_source[name]:
                s = next(i for i, line in enumerate(lines) if line.strip() == start.strip())
                e = next(i for i in range(s, len(lines)) if lines[i].strip() == end.strip())
                code = [line for line in lines[s:e + 1] if line.strip() and not _is_comment(line)]
                indent = len(code[0]) - len(code[0].lstrip(" "))
                lines = lines[:s] + [" " * indent + c for c in call.split("\n")] + lines[e + 1:]
            text = "\n".join(lines)
        assert current[name] == text, name
    added = set(current) - set(frozen)
    assert added == set(core.DESKTOP_HOOKS)
    for name, body in DESKTOP_HOOK_BODIES.items():
        assert _canonical_block(_function_body(current[name])) == _canonical_block(body.rstrip("\n").split("\n"))
    confirm = _canonical_block(_function_body(current["_confirm_clean_excluded_rectangle"]))
    frozen_clean = frozen["_handle_clean_this_rectangle"]
    asked = _canonical_block(_marker_block(frozen_clean, "reply = QMessageBox.question(",
                                           "print(f\"[CLEAN_RECT] User cancelled - rectangle {region_index} is excluded\")"))
    assert confirm == ["from PySide6.QtWidgets import QMessageBox"] + asked + ["    return False", "return True"]


def test_desktop_preview_only_calls_the_core():
    frozen = _methods(frozen_source("src/manga_image_preview.py"), "MangaImagePreviewWidget")
    current = _methods(current_source("manga_image_preview.py"), "MangaImagePreviewWidget")
    assert set(frozen) == set(current)
    for name in frozen:
        if name != "_on_stop_translation_clicked":
            assert current[name] == frozen[name], name
    lines = frozen["_on_stop_translation_clicked"].split("\n")
    for _helper, start, end_after, end, call in PREVIEW_SPLITS:
        s = next(i for i, line in enumerate(lines) if line.strip() == start.strip())
        a = next(i for i in range(s, len(lines)) if lines[i].strip() == end_after.strip())
        e = next(i for i in range(a, len(lines)) if lines[i].strip() == end.strip())
        indent = len(lines[s]) - len(lines[s].lstrip(" "))
        lines = lines[:s] + [" " * indent + c for c in call.split("\n")] + lines[e + 1:]
    assert current["_on_stop_translation_clicked"] == "\n".join(lines)
    # everything outside the widget class is untouched
    frozen_rest = {k: v for k, v in _top_defs(frozen_source("src/manga_image_preview.py")).items() if k != "MangaImagePreviewWidget"}
    current_rest = {k: v for k, v in _top_defs(current_source("manga_image_preview.py")).items() if k != "MangaImagePreviewWidget"}
    assert frozen_rest == current_rest


def test_manga_integration_reexports_the_state_manager():
    tree = ast.parse(current_source("manga_integration.py"))
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert "ImageStateManager" not in defined and "_state_manager_worker_process" not in defined
    imported = {(n.module, a.name) for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names}
    assert ("manga_editor_core", "ImageStateManager") in imported
    assert ("manga_editor_core", "_state_manager_worker_process") in imported


# ===========================================================================
# B / I: desktop binding and import hygiene
# ===========================================================================

def _free_globals(node):
    bound, loaded = set(), set()

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, n):
            bound.add(n.name)
            for a in n.args.args + n.args.kwonlyargs + n.args.posonlyargs:
                bound.add(a.arg)
            for extra in (n.args.vararg, n.args.kwarg):
                if extra:
                    bound.add(extra.arg)
            for d in n.decorator_list + n.args.defaults + [x for x in n.args.kw_defaults if x]:
                self.visit(d)
            for s in n.body:
                self.visit(s)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, n):
            for a in n.args.args + n.args.kwonlyargs:
                bound.add(a.arg)
            for extra in (n.args.vararg, n.args.kwarg):
                if extra:
                    bound.add(extra.arg)
            self.visit(n.body)

        def visit_ClassDef(self, n):
            bound.add(n.name)
            self.generic_visit(n)

        def visit_Name(self, n):
            (bound if isinstance(n.ctx, (ast.Store, ast.Del)) else loaded).add(n.id)

        def visit_Import(self, n):
            for a in n.names:
                bound.add((a.asname or a.name).split(".")[0])

        def visit_ImportFrom(self, n):
            for a in n.names:
                bound.add(a.asname or a.name)

        def visit_ExceptHandler(self, n):
            if n.name:
                bound.add(n.name)
            self.generic_visit(n)

    V().visit(node)
    return loaded - bound - set(dir(builtins))


@pytest.fixture(scope="module")
def image_renderer():
    pytest.importorskip("PySide6")
    import ImageRenderer
    return ImageRenderer


def test_desktop_runs_the_same_code_in_its_own_namespace(image_renderer):
    IR = image_renderer
    for name in core.EDITOR_FUNCTIONS:
        desktop, shared = getattr(IR, name), getattr(core, name)
        assert desktop is not shared
        assert desktop.__code__ is shared.__code__, name
        assert desktop.__globals__ is IR.__dict__, name
        assert desktop.__module__ == "ImageRenderer"
        assert desktop.__defaults__ == shared.__defaults__
    for name in core.SHARED_NAMES:
        assert getattr(IR, name) is getattr(core, name)
    for name in core.QT_HELPER_STUBS + core.DESKTOP_HOOKS:
        desktop = getattr(IR, name)
        assert desktop is not getattr(core, name)
        assert desktop.__code__.co_filename.endswith("ImageRenderer.py"), name
    tree = ast.parse(current_source("manga_editor_core.py"))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in core.EDITOR_FUNCTIONS:
        for free in _free_globals(funcs[name]):
            assert free in vars(IR), (name, free)
            assert free in vars(core), (name, free)


def test_qt_helper_stubs_cover_every_qt_call():
    """Every ImageRenderer function a moved body calls is moved itself, shared, or a stub."""
    frozen = _top_defs(frozen_source("src/ImageRenderer.py"))
    tree = ast.parse(current_source("manga_editor_core.py"))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    reached = set()
    for name in core.EDITOR_FUNCTIONS:
        reached |= {free for free in _free_globals(funcs[name]) if free in frozen}
    stubs = set(core.QT_HELPER_STUBS)
    assert reached - set(core.EDITOR_FUNCTIONS) == stubs


def test_core_imports_without_qt_or_the_desktop_modules():
    code = (
        "import sys\n"
        "for name in ('PySide6', 'shiboken6', 'tkinter', 'translator_gui', 'dpi_setup', 'ImageRenderer',\n"
        "             'manga_integration', 'manga_image_preview', 'manga_settings_dialog'):\n"
        "    sys.modules[name] = None\n"
        "import manga_editor_core\n"
        "heavy = [m for m in ('manga_translator', 'bubble_detector', 'local_inpainter', 'ocr_manager') if sys.modules.get(m)]\n"
        "assert not heavy, heavy\n"
        "print('ok')\n"
    )
    env = dict(os.environ, PYTHONPATH=str(SRC_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = subprocess.run([sys.executable, "-c", code], cwd=str(SRC_DIR), env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=600)
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-3000:]


def test_core_source_rules():
    raw = (SRC_DIR / "manga_editor_core.py").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.count(b"\r\n") in (0, raw.count(b"\n")), "mixed line endings"
    source = raw.decode("utf-8")
    tree = ast.parse(source, feature_version=(3, 10))
    banned = {"PySide6", "shiboken6", "translator_gui", "dpi_setup", "ImageRenderer", "manga_integration",
              "manga_image_preview", "manga_settings_dialog"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not {a.name.split(".")[0] for a in node.names} & banned, ast.dump(node)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned, ast.dump(node)


# ===========================================================================
# Fakes: the detector / OCR / LLM / inpainter the pipeline calls
# ===========================================================================

#: Detector boxes (x, y, w, h) on the 400x300 fixture page and what the fakes "read" in them.
BOXES = ((40, 40, 140, 80), (220, 170, 150, 90))
SOURCE_TEXT = {140: "konnichiwa", 150: "sekai"}           # by crop width


def _text_for_crop_width(width):
    """What the fake OCR reads in a crop: by its width, within a few pixels (the Qt preview's
    shapes are a pen width larger than the detector boxes)."""
    nearest = min(SOURCE_TEXT, key=lambda known: abs(known - width))
    return SOURCE_TEXT[nearest] if abs(nearest - width) <= 6 else ''
TRANSLATIONS = {"konnichiwa": "Hello there", "sekai": "World", "konbanwa": "Good evening"}
CALLS = []


class FakeDetector:
    def __init__(self, *args, **kwargs):
        pass

    def load_rtdetr_onnx_model(self, source):
        CALLS.append(("detector.load_rtdetr_onnx_model", os.path.basename(str(source))))
        return True

    def detect_with_rtdetr_onnx(self, image_path, confidence=0.3, return_all_bubbles=False):
        CALLS.append(("detector.detect_with_rtdetr_onnx", os.path.basename(image_path), confidence))
        return {'bubbles': [], 'text_bubbles': [BOXES[0]], 'text_free': [BOXES[1]]}


class FakeOCRResult:
    def __init__(self, text, bbox):
        self.text, self.bbox = text, bbox


class FakeOCRProvider:
    is_loaded = True

    def reset_stop_flags(self):
        pass


class FakeOCRManager:
    def __init__(self, log_callback=None):
        self.providers = {'custom-api': FakeOCRProvider(), 'rapidocr': FakeOCRProvider()}

    def get_provider(self, name):
        return self.providers.get(name)

    def load_provider(self, name, **kwargs):
        CALLS.append(("ocr.load_provider", name, sorted(kwargs)))
        return True

    def detect_text(self, image, provider, confidence=0.5, **kwargs):
        h, w = image.shape[:2]
        CALLS.append(("ocr.detect_text", provider, (h, w)))
        if provider == 'custom-api':
            return [FakeOCRResult(_text_for_crop_width(w), (0, 0, w, h))]
        # full-page providers: one line inside every detector box
        return [FakeOCRResult(SOURCE_TEXT[bw], (x + 5, y + 5, bw - 10, bh - 10)) for x, y, bw, bh in BOXES]

    def reset_stop_flags(self):
        pass


class FakeUnifiedClient:
    _cancelled = False

    def __init__(self, model=None, api_key=None, **kwargs):
        CALLS.append(("llm.client", model, bool(api_key)))

    @staticmethod
    def _model_needs_api_key(model):
        return True

    @staticmethod
    def _is_failed_finish_reason(reason):
        return reason in ('error', 'cancelled')

    @staticmethod
    def _is_api_error_placeholder(text):
        return False

    @classmethod
    def is_globally_cancelled(cls):
        return cls._cancelled

    @classmethod
    def set_global_cancellation(cls, value):
        cls._cancelled = bool(value)

    @classmethod
    def set_in_memory_multi_keys(cls, *args, **kwargs):
        CALLS.append(("llm.set_in_memory_multi_keys",))

    @classmethod
    def clear_in_memory_multi_keys(cls):
        CALLS.append(("llm.clear_in_memory_multi_keys",))

    @classmethod
    def setup_multi_key_pool(cls, *args, **kwargs):
        CALLS.append(("llm.setup_multi_key_pool",))

    @classmethod
    def set_in_memory_vision_keys(cls, *args, **kwargs):
        pass

    @classmethod
    def clear_in_memory_vision_keys(cls):
        pass

    send_hook = None

    def send(self, messages, temperature=None, max_tokens=None, **kwargs):
        text = str(messages[-1]['content']).strip().split("\n")[-1].strip()
        CALLS.append(("llm.send", text, temperature, max_tokens))
        if FakeUnifiedClient.send_hook:
            FakeUnifiedClient.send_hook(text)
        return TRANSLATIONS.get(text, "?" + text), "stop"

    def send_image(self, messages, image_base64, temperature=None, max_tokens=None, **kwargs):
        CALLS.append(("llm.send_image", len(image_base64)))
        return self.send(messages, temperature=temperature, max_tokens=max_tokens)


class FakeInpainter:
    model_loaded = True

    def __init__(self, *args, **kwargs):
        self.config = {}

    def inpaint(self, image, mask, iterations=None, _skip_hd=False, _skip_tiling=False):
        CALLS.append(("inpainter.inpaint", int(np.count_nonzero(mask)), iterations))
        out = image.copy()
        out[mask > 0] = (250, 250, 250)
        return out

    def set_log_callback(self, callback):
        self.log_callback = callback

    def download_jit_model(self, name):
        path = os.path.join(os.environ["HOME"], f"{name}.onnx")
        Path(path).write_bytes(b"fake model")
        return path

    def load_model(self, *args, **kwargs):
        return True

    def reset_stop_flags(self):
        pass


def _draw_regions(image_bgr, regions):
    out = image_bgr.copy()
    for region in regions:
        x, y, w, h = [int(v) for v in region.bounding_box]
        shade = 40 + (sum(map(ord, str(region.translated_text or ''))) % 150)
        cv2.rectangle(out, (x + 4, y + 4), (x + w - 4, y + h - 4), (shade, 0, 255 - shade), -1)
    return out


class FakeMangaTranslator:
    _inpaint_pool = {}
    _inpaint_pool_lock = threading.Lock()
    _cancelled = False

    @classmethod
    def reset(cls):
        cls._inpaint_pool = {}
        cls._cancelled = False

    @classmethod
    def is_globally_cancelled(cls):
        return cls._cancelled

    @classmethod
    def set_global_cancellation(cls, value):
        cls._cancelled = bool(value)

    @classmethod
    def reset_global_flags(cls):
        cls._cancelled = False

    @classmethod
    def force_release_all_pool_checkouts(cls):
        return (0, 0)

    @classmethod
    def hard_cancel_all(cls):
        pass

    def __init__(self, ocr_config=None, unified_client=None, main_gui=None, log_callback=None, skip_inpainter_init=False):
        CALLS.append(("mt.init", (ocr_config or {}).get('provider'), skip_inpainter_init))
        self.settings = {}

    def _get_thread_bubble_detector(self):
        return FakeDetector()

    def _return_bubble_detector_to_pool(self):
        CALLS.append(("mt.return_detector",))

    def _get_or_init_shared_local_inpainter(self, method, model_path, force_reload=False):
        CALLS.append(("mt.shared_inpainter", method, os.path.basename(model_path or '')))
        return FakeInpainter()

    def _return_inpainter_to_pool(self):
        CALLS.append(("mt.return_inpainter",))

    def preload_local_inpainters_concurrent(self, method, model_path, count, max_parallel=None):
        return 1

    def translate_full_page_context(self, regions, image_path):
        CALLS.append(("mt.full_page", [r.text for r in regions]))
        for region in regions:
            region.translated_text = TRANSLATIONS.get(region.text, "?" + region.text)
        return {f"[{i}] {r.text}": r.translated_text for i, r in enumerate(regions)}

    def update_text_rendering_settings(self, **kwargs):
        self.settings = dict(kwargs)

    def render_translated_text(self, image_bgr, regions):
        CALLS.append(("mt.render", [(tuple(r.bounding_box), r.translated_text) for r in regions]))
        return _draw_regions(image_bgr, regions)

    def restore_print(self):
        pass


class _TextField:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class FakeMainGui:
    """The main_gui surface the editor functions read (HeadlessOwner provides the same)."""

    def __init__(self, config):
        self.config = config
        self.api_key_entry = _TextField(config.get('api_key', ''))
        self.model_var = config.get('model', 'gpt-test')

    def _get_environment_variables(self, epub_path='', api_key=''):
        return {'MODEL': self.model_var, 'SYSTEM_PROMPT': 'translation system prompt', 'MANGA_TRACE_ENV': '1'}


#: The manga tab's values the editor functions read (MangaTranslationTab._load_rendering_settings).
TAB_VALUES = {
    'inpaint_method_value': 'local', 'local_model_type_value': 'anime_onnx', 'local_model_path_value': '',
    'skip_inpainting_value': False, 'inpaint_quality_value': 'high', 'inpaint_dilation_value': 15,
    'inpaint_passes_value': 2, 'bg_opacity_value': 0, 'free_text_only_bg_opacity_value': False,
    'bg_style_value': 'circle', 'bg_reduction_value': 1.0, 'font_size_value': 0, 'font_size_mode_value': 'fixed',
    'font_size_multiplier_value': 1.0, 'auto_min_size_value': 10, 'max_font_size_value': 36,
    'force_caps_lock_value': True, 'constrain_to_bubble_value': True, 'strict_text_wrapping_value': True,
    'safe_area_enabled_value': False, 'safe_area_scale_value': 1.0, 'text_color_r_value': 102,
    'text_color_g_value': 0, 'text_color_b_value': 0, 'shadow_enabled_value': False,
    'shadow_color_r_value': 255, 'shadow_color_g_value': 255, 'shadow_color_b_value': 255,
    'shadow_offset_x_value': 2, 'shadow_offset_y_value': 2, 'shadow_blur_value': 0,
    'font_style_value': 'Default', 'full_page_context_value': False,
}

BASE_CONFIG = {
    'api_key': 'test-key', 'model': 'gpt-test', 'manga_ocr_provider': 'custom-api',
    'manga_inpaint_method': 'opencv', 'manga_custom_api_ocr_batch_enabled': False,
    'system_prompt': 'Translate to English.', 'manga_settings': {'inpainting': {'method': 'local'}},
}


def make_page(path, labels=("konnichiwa", "sekai")):
    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT_PATH), 18) if FONT_PATH.exists() else ImageFont.load_default()
    for (x, y, w, h), label in zip(BOXES, labels):
        draw.ellipse((x - 10, y - 10, x + w + 10, y + h + 10), outline="black", width=3)
        draw.text((x + 15, y + h // 2 - 10), label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path))
    return str(path)


class TracePreview:
    """The preview surface (current page, translated output, viewer) without widgets."""

    def __init__(self, page, boxes=()):
        self.current_image_path = page
        self.current_translated_path = None
        self.image_paths = [page] if page else []
        self.source_display_mode = 'translated'
        self.cleaned_images_enabled = True
        self.viewer = core.EditorViewer()
        self.viewer.rectangles.extend(boxes)
        self.loads = []

    def load_image(self, path, preserve_rectangles=False, preserve_text_overlays=False):
        self.loads.append((path, preserve_rectangles, preserve_text_overlays))
        self.current_image_path = path


class TraceHost:
    """The manga tab surface the moved functions read (identical for the frozen, desktop and
    mobile namespaces)."""

    def __init__(self, root, config, page, boxes=()):
        self.main_gui = FakeMainGui(json.loads(json.dumps(config)))
        self.update_queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.logs = []
        self.image_state_manager = core.ImageStateManager(str(root / "state" / "image_state.json"))
        self.image_preview_widget = TracePreview(page, boxes)
        self._use_circle_shapes = False
        self.translator = None
        self._manga_translator = None
        self.ocr_manager = None
        self._shared_inpainter = None
        self.ocr_prompt = 'OCR SYSTEM PROMPT'
        self._current_image_path = page
        self.dialog = None
        for name, value in TAB_VALUES.items():
            setattr(self, name, value)
        if FONT_PATH.exists():
            self.selected_font_path = str(FONT_PATH)

    def _log(self, message, level='info'):
        self.logs.append((level, str(message)))

    def _default_manga_ocr_prompt(self):
        return 'DEFAULT OCR PROMPT'

    def drain(self):
        items = []
        while True:
            try:
                items.append(self.update_queue.get_nowait())
            except queue.Empty:
                return items


def _boxes(*specs):
    out = []
    for i, (x, y, w, h) in enumerate(specs):
        box = core.EditorBox(x, y, w, h, region_index=i, bubble_type='text_bubble', region_type='text_bubble')
        out.append(box)
    return out


def _digest_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() + f":{image.shape}"


def _normalize(value, roots):
    if isinstance(value, dict):
        return {_normalize(k, roots): _normalize(v, roots) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v, roots) for v in value]
    if isinstance(value, np.ndarray):
        return "ndarray:" + hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    if isinstance(value, str):
        for root in roots:
            for form in {root, root.replace("\\", "/"), root.replace("/", "\\")}:
                value = value.replace(form, "<ROOT>")
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if hasattr(value, 'bounding_box'):
        return ("TextRegion", tuple(value.bounding_box), value.text, getattr(value, 'translated_text', None))
    return repr(type(value).__name__)


def _outputs(root, inputs):
    found = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and str(path) not in inputs and path.suffix.lower() in ('.png', '.jpg', '.json'):
            if path.suffix.lower() == '.json':
                continue
            found[path.relative_to(root).as_posix()] = _digest_image(path)
    return found


@functools.lru_cache(maxsize=None)
def legacy_image_renderer():
    """ImageRenderer at U8_BASE_SHA as a module (the frozen desktop oracle)."""
    pytest.importorskip("PySide6")
    module = types.ModuleType("legacy_ImageRenderer_u8")
    module.__file__ = str(SRC_DIR / "ImageRenderer.py")
    sys.modules[module.__name__] = module
    exec(compile(frozen_source("src/ImageRenderer.py"), "<ImageRenderer @ U8 base>", "exec"), module.__dict__)
    return module


def _namespace(side):
    if side == "legacy":
        return legacy_image_renderer()
    if side == "desktop":
        pytest.importorskip("PySide6")
        import ImageRenderer
        return ImageRenderer
    return core


SIDES = ("legacy", "desktop", "mobile")

_LOOPBACK = ("127.0.0.1", "::1", "localhost")


def _block_remote_network(monkeypatch):
    """No test reaches a real API: only loopback connections (the fake Vision server) are allowed."""
    import socket

    def _host(address):
        return address[0] if isinstance(address, tuple) and address else address

    def guarded_getaddrinfo(host, *args, _real=socket.getaddrinfo, **kwargs):
        if host not in _LOOPBACK and host is not None:
            raise OSError(f"network access blocked in tests: {host}")
        return _real(host, *args, **kwargs)

    def guarded_connect(self, address, _real=socket.socket.connect):
        if isinstance(address, tuple) and _host(address) not in _LOOPBACK:
            raise OSError(f"network access blocked in tests: {_host(address)}")
        return _real(self, address)

    def guarded_connect_ex(self, address, _real=socket.socket.connect_ex):
        if isinstance(address, tuple) and _host(address) not in _LOOPBACK:
            raise OSError(f"network access blocked in tests: {_host(address)}")
        return _real(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


def _loaded_editor_namespaces():
    """The three namespaces the moved functions run in (frozen / desktop only with PySide6)."""
    namespaces = [core]
    try:
        import PySide6  # noqa: F401
    except ImportError:
        return namespaces
    import ImageRenderer
    return namespaces + [ImageRenderer, legacy_image_renderer()]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Scratch HOME / Library / output dirs, no worker processes, and the process-wide stop
    flags the editor functions flip restored afterwards."""
    saved = dict(os.environ)
    home = tmp_path / "_home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "_library"))
    monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")  # no http_requests/ traces next to src (mobile default)
    monkeypatch.delenv("OUTPUT_DIRECTORY", raising=False)
    monkeypatch.delenv("GLOSSARION_DATA_DIR", raising=False)
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "_tmp"))
    (tmp_path / "_tmp").mkdir()
    _block_remote_network(monkeypatch)
    yield
    monkeypatch.undo()
    os.environ.clear()
    os.environ.update(saved)
    for module_name, call in (("manga_translator", "MangaTranslator.set_global_cancellation"),
                              ("unified_api_client", "set_stop_flag"),
                              ("unified_api_client", "UnifiedClient.set_global_cancellation"),
                              ("TransateKRtoEN", "set_stop_flag")):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        target = module
        try:
            for part in call.split("."):
                target = getattr(target, part)
            target(False)
        except Exception:
            pass


@pytest.fixture
def fakes(monkeypatch):
    import bubble_detector
    import local_inpainter
    import manga_translator
    import ocr_manager
    import unified_api_client

    CALLS.clear()
    FakeMangaTranslator.reset()
    FakeUnifiedClient._cancelled = False
    FakeUnifiedClient.send_hook = None
    monkeypatch.setattr(manga_translator, "MangaTranslator", FakeMangaTranslator)
    monkeypatch.setattr(unified_api_client, "UnifiedClient", FakeUnifiedClient)
    monkeypatch.setattr(ocr_manager, "OCRManager", FakeOCRManager)
    monkeypatch.setattr(local_inpainter, "LocalInpainter", FakeInpainter)
    monkeypatch.setattr(bubble_detector, "BubbleDetector", FakeDetector)
    # ImageRenderer (and so the editor namespaces) binds UnifiedClient at import time
    for namespace in _loaded_editor_namespaces():
        monkeypatch.setattr(namespace, "UnifiedClient", FakeUnifiedClient)
    return CALLS


# ===========================================================================
# S: the per-image state manager
# ===========================================================================

class _FakeMpContext:
    def __init__(self):
        self.processes = []

    def Queue(self):
        return queue.Queue()

    def Process(self, target=None, args=(), daemon=None):
        record = types.SimpleNamespace(target=target, args=args, daemon=daemon, started=False, alive=False)

        def start():
            record.started = record.alive = True

        record.start = start
        record.is_alive = lambda: record.alive
        record.join = lambda timeout=None: None

        def terminate():
            record.alive = False

        record.terminate = terminate
        self.processes.append(record)
        return record


def _spawn_tripwire(*args, **kwargs):
    raise AssertionError("multiprocessing used while processes are unavailable")


def test_state_manager_runs_in_process_on_mobile(tmp_path, monkeypatch):
    import multiprocessing
    monkeypatch.setattr(multiprocessing, "get_context", _spawn_tripwire)
    monkeypatch.setattr(multiprocessing, "Process", _spawn_tripwire)
    state_file = tmp_path / "state" / "image_state.json"
    manager = core.ImageStateManager(str(state_file))
    assert manager._mp_enabled is False and manager._mp_worker is None and manager._mp_task_q is None
    manager.update_state("C:\\pages\\001.png", {'detection_regions': [{'bbox': [1, 2, 3, 4]}]})
    manager.set_state("/pages/002.png", {'recognized_texts': [{'text': 'a'}], 'excluded_from_clean': [0]})
    assert manager.get_state("C:/pages/001.png")['detection_regions'][0]['bbox'] == [1, 2, 3, 4]
    assert 'excluded_from_clean' not in manager.get_state("/pages/002.png")
    manager.flush()
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert set(saved) == {"C:/pages/001.png", "/pages/002.png"}
    reloaded = core.ImageStateManager(str(state_file))
    assert reloaded.get_state("C:/pages/001.png") == saved["C:/pages/001.png"]
    reloaded.clear_state("C:/pages/001.png")
    reloaded.flush_async()
    reloaded._async_flush_thread.join(5)
    assert set(json.loads(state_file.read_text(encoding="utf-8"))) == {"/pages/002.png"}


def test_state_manager_keeps_the_desktop_worker(tmp_path, monkeypatch):
    import multiprocessing
    monkeypatch.delenv("GLOSSARION_NO_PROCESSES", raising=False)
    monkeypatch.delenv("GLOSSARION_MOBILE", raising=False)
    monkeypatch.delenv("FLET_PLATFORM", raising=False)
    context = _FakeMpContext()
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: context if method == 'spawn' else None)
    manager = core.ImageStateManager(str(tmp_path / "state.json"))
    assert manager._mp_enabled is True
    assert len(context.processes) == 1
    process = context.processes[0]
    assert process.started and process.daemon is True
    assert process.target is core._state_manager_worker_process
    assert process.args[2] == str(tmp_path / "state.json")
    manager.update_state("p.png", {'step': 'detected'})
    assert manager._mp_task_q.get_nowait() == {'type': 'update_state', 'image_path': 'p.png', 'updates': {'step': 'detected'}}
    manager._stop_worker()
    assert manager._mp_enabled is False


def test_state_manager_matches_the_frozen_class(tmp_path, monkeypatch):
    """Same operations on the frozen manga_integration ImageStateManager and the moved one (both
    with a recording multiprocessing context): same task messages, same files."""
    import multiprocessing
    from typing import Any, Dict, Optional
    from manga_files_core import _manga_cmd_debug_print

    monkeypatch.delenv("GLOSSARION_NO_PROCESSES", raising=False)
    namespace = {'os': os, 'json': json, 'threading': threading, 'Dict': Dict, 'Any': Any, 'Optional': Optional,
                 '_manga_cmd_debug_print': _manga_cmd_debug_print, '__name__': 'frozen_state_manager'}
    exec(compile(_frozen_state_block(), "<frozen ImageStateManager>", "exec"), namespace)
    results = {}
    for side, cls in (("frozen", namespace['ImageStateManager']), ("moved", core.ImageStateManager)):
        context = _FakeMpContext()
        monkeypatch.setattr(multiprocessing, "get_context", lambda method, c=context: c)
        path = tmp_path / side / "state.json"
        manager = cls(str(path))
        manager.update_state("a\\001.png", {'detection_regions': [1]})
        manager.set_state("a/002.png", {'recognized_texts': [{'text': 'x'}], 'excluded_from_clean': [1]})
        manager.get_state("a/002.png")
        manager.clear_state("a/001.png")
        manager.update_state("a/003.png", {'translated_texts': []}, save=False)
        manager.flush()
        tasks = []
        while True:
            try:
                tasks.append(manager._mp_task_q.get_nowait())
            except queue.Empty:
                break
        results[side] = (tasks, json.loads(path.read_text(encoding="utf-8")))
        manager._stop_worker()
    assert results["frozen"] == results["moved"]


# ===========================================================================
# T: trace parity of the per-image pipeline (frozen vs desktop binding vs mobile namespace)
# ===========================================================================

TRACKED_ATTRS = ('_current_regions', '_recognition_data', '_translation_data', '_recognized_texts',
                 '_translated_texts', '_cleaned_image_path', '_rendered_images_map', '_batch_mode_active')


def _scenario_detect(M, host, pages):
    config = M._get_detection_config(host)
    config['detect_empty_bubbles'] = False
    M._run_detect_background(host, pages[0], config)


def _scenario_recognize(M, host, pages):
    M._run_recognize_background(host, pages[0], None, M._get_ocr_config(host))


def _scenario_recognize_full_page_provider(M, host, pages):
    host.main_gui.config['manga_ocr_provider'] = 'rapidocr'
    M._run_recognize_background(host, pages[0], [{'bbox': list(b), 'confidence': 0.9} for b in BOXES],
                                M._get_ocr_config(host))


def _scenario_clean_opencv(M, host, pages):
    host.image_preview_widget.viewer.rectangles[1].exclude_from_clean = True
    regions = M._extract_regions_from_preview(host)
    M._run_clean_background(host, pages[0], regions)


def _scenario_clean_local_pool(M, host, pages):
    host.main_gui.config['manga_inpaint_method'] = 'local'
    host.main_gui.config['manga_local_inpaint_model'] = 'anime_onnx'
    host.image_preview_widget.viewer.rectangles[0].inpaint_iterations = 4
    M._run_clean_background(host, pages[0], M._extract_regions_from_preview(host))


def _scenario_translate_pipeline(M, host, pages):
    host.skip_inpainting_value = True
    M._run_full_translate_pipeline(host, pages[0], None)


def _scenario_translate_full_page_context(M, host, pages):
    host.skip_inpainting_value = True
    host.full_page_context_value = True
    M._run_full_translate_pipeline(host, pages[0], [{'bbox': list(b), 'confidence': 1.0} for b in BOXES])


def _scenario_translate_with_inpainting(M, host, pages):
    M._run_full_translate_pipeline(host, pages[0], None)


def _scenario_translate_all(M, host, pages):
    host.image_state_manager.update_state(pages[1], manga_ocr_io_state(pages[1]))
    host._batch_full_page_context_enabled = False
    host._batch_visual_context_enabled = False
    M._run_translate_all_background(host, list(pages))


def _scenario_translate_this_text(M, host, pages):
    host._recognition_data = {1: {'text': 'sekai', 'bbox': list(BOXES[1])}}
    # the frozen desktop builds this prompt inline in its context menu (split out as core._manual_translate_prompt)
    prompt, _language = core._manual_translate_prompt(host)
    M._translate_this_text_background(host, f"{prompt}\n\nsekai", 1)


def _scenario_page_state(M, host, pages):
    page = pages[0]
    host.image_state_manager.set_state(page, {
        'recognized_texts': [{'text': 'konnichiwa', 'bbox': list(BOXES[0]), 'region_index': 0}, {'deleted': True}],
        'translated_texts': [{'original': {'text': 'konnichiwa', 'region_index': 0}, 'translation': 'Hello there',
                              'bbox': list(BOXES[0])}],
        'cleaned_image_path': str(Path(page).with_name("missing_cleaned.png")),
    })
    M._validate_and_clean_stale_state(host, page)
    counts = M._rehydrate_text_state_from_persisted(host, page)
    host.logs.append(('trace', repr(counts)))
    host._current_regions = [{'bbox': list(b)} for b in BOXES]
    host._cleaned_image_path = None
    M._persist_current_image_state(host)
    M._clear_cross_image_state(host)
    M._clear_detection_state_for_image(host, page)


def manga_ocr_io_state(page):
    import manga_ocr_io
    return manga_ocr_io.editor_state_from_page({'regions': [
        {'bbox': [40, 40, 140, 80], 'text': 'konbanwa', 'bubble_type': 'text_bubble'},
    ]})


SCENARIOS = {
    'detect': (_scenario_detect, False),
    'recognize': (_scenario_recognize, False),
    'recognize_full_page_provider': (_scenario_recognize_full_page_provider, False),
    'clean_opencv': (_scenario_clean_opencv, False),
    'clean_local_pool': (_scenario_clean_local_pool, False),
    'translate_pipeline': (_scenario_translate_pipeline, False),
    'translate_full_page_context': (_scenario_translate_full_page_context, False),
    'translate_with_inpainting': (_scenario_translate_with_inpainting, True),
    'translate_all': (_scenario_translate_all, True),
    'translate_this_text': (_scenario_translate_this_text, False),
    'page_state': (_scenario_page_state, False),
}


def _run_trace(tmp_path, side, scenario):
    run, unordered = SCENARIOS[scenario]
    root = tmp_path / side
    pages = [make_page(root / "001.png"), make_page(root / "002.png", labels=("konbanwa", "sekai"))]
    host = TraceHost(root, BASE_CONFIG, pages[0], boxes=_boxes(*BOXES))
    CALLS.clear()
    FakeMangaTranslator.reset()
    run(_namespace(side), host, pages)
    messages = host.drain()
    logs = host.logs
    calls = list(CALLS)
    if unordered:
        messages = sorted(messages, key=repr)
        logs = sorted(logs)
        calls = sorted(calls, key=repr)
    trace = {
        'queue': messages,
        'logs': logs,
        'calls': calls,
        'state': host.image_state_manager._states,
        'attrs': {name: getattr(host, name, '<unset>') for name in TRACKED_ATTRS},
        'boxes': [b.to_dict() for b in host.image_preview_widget.viewer.rectangles],
        'outputs': _outputs(root, set(pages)),
    }
    return _normalize(trace, [str(root)])


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_pipeline_trace_matches_the_frozen_desktop(scenario, tmp_path, fakes, monkeypatch):
    if scenario == 'translate_all':
        monkeypatch.setattr(time, "sleep", lambda seconds: None)
    traces = {side: _run_trace(tmp_path, side, scenario) for side in SIDES}
    assert traces["desktop"] == traces["legacy"]
    assert traces["mobile"] == traces["legacy"]
    assert traces["legacy"]['queue'] or traces["legacy"]['state'] or traces["legacy"]['logs']


# ===========================================================================
# R: image-diff of the PIL re-render on a fixture page with a fixed font
# ===========================================================================

import manga_translator as _manga_translator_module  # noqa: E402

#: The real class (the trace fixtures swap ``manga_translator.MangaTranslator`` for a fake).
REAL_MANGA_TRANSLATOR = _manga_translator_module.MangaTranslator


@functools.lru_cache(maxsize=None)
def _real_manga_translator():
    """One real render-only MangaTranslator (the PIL renderer under test), shared by all sides."""
    return REAL_MANGA_TRANSLATOR(ocr_config={'provider': 'custom-api'},
                                 unified_client=types.SimpleNamespace(model='gpt-test'),
                                 main_gui=types.SimpleNamespace(config={}), log_callback=None,
                                 skip_inpainter_init=True)


def _render_host(root, side_boxes=None):
    page = make_page(root / "001.png")
    host = TraceHost(root, BASE_CONFIG, page, boxes=side_boxes or _boxes(*BOXES))
    host._manga_translator = _real_manga_translator()
    state = {
        'recognized_texts': [{'text': 'konnichiwa', 'bbox': list(BOXES[0]), 'region_index': 0},
                             {'text': 'sekai', 'bbox': list(BOXES[1]), 'region_index': 1}],
        'translated_texts': [
            {'original': {'text': 'konnichiwa', 'region_index': 0}, 'translation': 'Hello there', 'bbox': list(BOXES[0])},
            {'original': {'text': 'sekai', 'region_index': 1}, 'translation': 'World', 'bbox': list(BOXES[1])},
        ],
        'last_render_positions': {'1': [230, 180, 130, 70]},
    }
    host.image_state_manager.set_state(page, state)
    host._translation_data = {0: {'original': 'konnichiwa', 'translation': 'Hello there'},
                              1: {'original': 'sekai', 'translation': 'World'}}
    host._translated_texts = state['translated_texts']
    return host, page


def _render_regions(texts):
    from manga_translator import TextRegion
    regions = []
    for (x, y, w, h), text in zip(BOXES, texts):
        region = TextRegion(text='src', vertices=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                            bounding_box=(x, y, w, h), confidence=1.0, region_type='text_block')
        region.translated_text = text
        regions.append(region)
    return regions


def _render_case_direct(M, host, page):
    host.image_preview_widget.viewer.rectangles[1].exclude_from_clean = True
    out = str(Path(page).parent / "001_translated" / "001.png")
    return M._render_with_manga_translator(host, page, _render_regions(["Hello there", "World"]), output_path=out,
                                           original_image_path=page, switch_tab=False, refresh_preview=False)


def _render_case_save_positions(M, host, page):
    host.image_preview_widget.viewer.rectangles[0].set_geometry(60, 60, 150, 90)
    host.image_state_manager.get_state(page)['last_render_positions'] = {}
    M.save_positions_and_rerender(host)
    return host.image_state_manager.get_state(page).get('rendered_image_path')


def _render_case_persisted(M, host, page):
    return M.render_persisted_translation_state(host, page, refresh_preview=False)


def _render_case_single_overlay(M, host, page):
    host._translating_image_path = page
    host.image_preview_widget.viewer.rectangles[1].set_geometry(200, 150, 180, 120)
    M._update_single_text_overlay(host, 1, 'World')
    return host.image_state_manager.get_state(page).get('rendered_image_path')


RENDER_CASES = {
    'direct': _render_case_direct,
    'save_positions': _render_case_save_positions,
    'persisted_state': _render_case_persisted,
    'single_overlay': _render_case_single_overlay,
}


@pytest.mark.parametrize("case", sorted(RENDER_CASES))
def test_rerender_image_matches_the_frozen_desktop(case, tmp_path):
    assert FONT_PATH.exists(), ("the fixed test font is missing: whitelist tests/fixtures/manga_editor/ "
                                "in .gitignore so CI checks it out")
    results = {}
    for side in SIDES:
        root = tmp_path / side
        host, page = _render_host(root)
        output = RENDER_CASES[case](_namespace(side), host, page)
        assert output and os.path.isfile(output), (side, output, host.logs[-3:])
        rendered = cv2.imread(output)
        source = cv2.imread(page)
        results[side] = (
            _normalize(output, [str(root)]),
            hashlib.sha256(rendered.tobytes()).hexdigest(),
            _normalize(host.image_state_manager.get_state(page), [str(root)]),
            [entry for entry in host.logs if entry[0] != 'debug'],
        )
        assert rendered.shape == source.shape
        assert int(np.count_nonzero(np.any(rendered != source, axis=2))) > 200, "nothing was rendered"
    assert results["desktop"] == results["legacy"]
    assert results["mobile"] == results["legacy"]


# ===========================================================================
# Q: offscreen smoke of the desktop preview editor actions (frozen vs rewired ImageRenderer)
# ===========================================================================

@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _qt_dispatch(M, tab, update):
    kind = update[0]
    data = update[1] if len(update) > 1 else None
    if kind == 'detect_results':
        M._process_detect_results(tab, data)
    elif kind == 'recognize_results':
        M._process_recognize_results(tab, data)
    elif kind == 'translate_results':
        M._process_translate_results(tab, data)
    elif kind == 'single_clean_complete':
        M._update_image_preview_with_result(tab, data['result_image'], data['original_path'])
    elif kind == 'preview_update':
        tab.image_preview_widget.current_translated_path = data.get('translated_path')
    elif kind == 'call_method':
        update[1](*update[2])
    elif kind in ('detect_button_restore', 'recognize_button_restore', 'translate_button_restore',
                  'clean_button_restore'):
        getattr(M, '_restore_' + kind[:-len('_button_restore')] + '_button')(tab)


def _qt_pump(app, M, tab, names, before):
    suffixes = tuple(f"({name})" for name in names)
    deadline = time.time() + 120
    while time.time() < deadline:
        app.processEvents()
        alive = any(t.is_alive() and t not in before and t.name.endswith(suffixes) for t in threading.enumerate())
        try:
            update = tab.update_queue.get(timeout=0.02)
        except queue.Empty:
            if not alive:
                break
            continue
        _qt_dispatch(M, tab, update)
    for _ in range(5):
        app.processEvents()
        time.sleep(0.01)


def _qt_action(app, M, tab, action, args=(), names=()):
    before = set(threading.enumerate())
    action(tab, *args)
    _qt_pump(app, M, tab, names, before)


def _qt_teardown(app, tab, widget, M=None):
    """Delete a preview widget safely: let the editor's delayed refresh timers fire while it still
    exists (ImageRenderer's longest QTimer.singleShot is 3000 ms), stop the looping pulse
    animations (they paint QGraphicsItems of its scene), then delete it and flush the deferred
    deletes, so no later event loop (pytest-qt's, another test's) touches a deleted item."""
    from PySide6.QtCore import QAbstractAnimation, QCoreApplication, QEvent
    import gc

    deadline = time.monotonic() + 3.5
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)
    animations = [entry.get('animation') for entry in (getattr(tab, '_processing_overlays_by_image', None) or {}).values()
                  if isinstance(entry, dict)]
    animations += [getattr(item, '_pulse_animation', None) for item in list(widget.viewer.rectangles)]
    animations += [obj for obj in gc.get_objects() if isinstance(obj, QAbstractAnimation)]
    for animation in animations:
        try:
            if animation is not None:
                animation.stop()
        except Exception:
            pass
    if M is not None:
        try:
            M._remove_processing_overlay(tab, clear_all=True)
        except Exception:
            pass
    timer = getattr(tab, '_output_refresh_timer', None)
    if timer is not None:
        try:
            timer.stop()
        except Exception:
            pass
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def _wait_loaded(app, widget):
    deadline = time.time() + 20
    while not widget.viewer.hasPhoto() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert widget.viewer.hasPhoto()


def _qt_snapshot(tab, root):
    rects = []
    for item in tab.image_preview_widget.viewer.rectangles:
        r = item.sceneBoundingRect()
        rects.append((round(r.x(), 2), round(r.y(), 2), round(r.width(), 2), round(r.height(), 2),
                      getattr(item, 'region_index', None), bool(getattr(item, 'is_recognized', False)),
                      bool(getattr(item, 'exclude_from_clean', False)), getattr(item, 'inpaint_iterations', None),
                      getattr(item, 'bubble_type', None)))
    page = tab.image_preview_widget.current_image_path
    return _normalize({
        'rects': rects,
        'recognition': getattr(tab, '_recognition_data', None),
        'translation': getattr(tab, '_translation_data', None),
        'state': tab.image_state_manager._states,
        'outputs': _outputs(root, {page}),
        'logs': [entry for entry in tab.logs if entry[0] != 'debug'],
    }, [str(root)])


def _patched_dialog_exec(monkeypatch, edits):
    from PySide6 import QtWidgets

    def fake_exec(dialog):
        texts = dialog.findChildren(QtWidgets.QTextEdit)
        for edit, value in zip(texts, edits()):
            if value is not None:
                edit.setPlainText(value)
        save = [b for b in dialog.findChildren(QtWidgets.QPushButton) if b.objectName() == 'save_btn'][0]
        save.click()
        return 1

    monkeypatch.setattr(QtWidgets.QDialog, "exec", fake_exec)


def _qt_session(app, side, tmp_path, monkeypatch):
    from PySide6 import QtWidgets
    import manga_image_preview

    M = _namespace(side)
    root = tmp_path / side
    page = make_page(root / "001.png")
    widget = manga_image_preview.MangaImagePreviewWidget(main_gui=types.SimpleNamespace(config={}))
    tab = TraceHost(root, BASE_CONFIG, page)
    tab.image_preview_widget = widget
    tab._rendering_in_progress = False
    tab._manga_translator = FakeMangaTranslator()
    tab.ocr_result_signal = core._QueuedSignal(tab.update_queue, lambda *a: M._process_ocr_result(tab, *a))
    tab.ocr_error_signal = core._QueuedSignal(tab.update_queue, lambda *a: M._handle_ocr_error(tab, *a))
    widget.manga_integration = tab
    widget.load_image(page)
    _wait_loaded(app, widget)
    snapshots = {}

    _qt_action(app, M, tab, M._on_detect_text_clicked, names=('_run_detect_background',))
    snapshots['detect'] = _qt_snapshot(tab, root)
    _qt_action(app, M, tab, M._on_recognize_text_clicked, names=('_run_recognize_background',))
    snapshots['recognize'] = _qt_snapshot(tab, root)
    tab.skip_inpainting_value = True
    _qt_action(app, M, tab, M._on_translate_text_clicked,
               names=('_run_translate_background', '_run_full_translate_pipeline', 'run_inpainting_concurrent'))
    snapshots['translate'] = _qt_snapshot(tab, root)

    _patched_dialog_exec(monkeypatch, lambda: ["konnichiwa!"])
    M._show_ocr_popup(tab, tab._recognition_data[0]['text'], 0)
    snapshots['edit_ocr'] = _qt_snapshot(tab, root)
    _patched_dialog_exec(monkeypatch, lambda: [None, "Hi there"])
    M._show_translation_popup(tab, tab._translation_data[0], 0)
    for _ in range(20):
        app.processEvents()
        time.sleep(0.01)
    snapshots['edit_translation'] = _qt_snapshot(tab, root)

    rects = widget.viewer.rectangles
    monkeypatch.setattr(QtWidgets.QInputDialog, "getInt", staticmethod(lambda *a, **k: (7, True)))
    M._handle_set_inpainting_iterations(tab, 1, rects[1])
    M._handle_toggle_free_text_region(tab, 0, rects[0])
    snapshots['box_flags'] = _qt_snapshot(tab, root)

    _qt_action(app, M, tab, M._handle_clean_this_rectangle, (1, rects[1]), names=('run_single_rect_clean',))
    snapshots['clean_rect'] = _qt_snapshot(tab, root)
    rects[0].exclude_from_clean = True
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.No))
    _qt_action(app, M, tab, M._handle_clean_this_rectangle, (0, rects[0]), names=('run_single_rect_clean',))
    snapshots['clean_excluded'] = _qt_snapshot(tab, root)

    M._handle_delete_rectangle(tab, 1, rects[1])
    snapshots['delete'] = _qt_snapshot(tab, root)
    _qt_teardown(app, tab, widget, M)
    return snapshots


def test_preview_editor_actions_match_the_frozen_desktop(qapp, tmp_path, fakes, monkeypatch):
    legacy = _qt_session(qapp, "legacy", tmp_path, monkeypatch)
    current = _qt_session(qapp, "desktop", tmp_path, monkeypatch)
    for step in legacy:
        assert current[step] == legacy[step], step
    assert legacy['detect']['rects'] and all(r[5] for r in legacy['recognize']['rects'])
    assert legacy['edit_ocr']['recognition'][0]['text'] == 'konnichiwa!'
    assert legacy['edit_translation']['translation'][0]['translation'] == 'Hi there'
    assert legacy['box_flags']['rects'][1][7] == 7 and legacy['box_flags']['rects'][0][8] == 'free_text'
    assert any(name.endswith('_cleaned.png') for name in legacy['clean_rect']['outputs'])
    assert len(legacy['delete']['rects']) == 1


def test_preview_stop_button_sets_the_editor_flags(qapp, tmp_path, fakes, monkeypatch):
    import manga_image_preview

    widget = manga_image_preview.MangaImagePreviewWidget(main_gui=types.SimpleNamespace(config={}))
    tab = TraceHost(tmp_path, BASE_CONFIG, make_page(tmp_path / "001.png"))
    tab.is_running = True
    tab._batch_mode_active = True
    tab._global_cancellation = False
    tab.set_global_cancellation = lambda value: setattr(tab, '_forced', value)
    widget.manga_integration = tab
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    widget._on_stop_translation_clicked()
    assert os.environ['GRACEFUL_STOP'] == '1' and os.environ['WAIT_FOR_CHUNKS'] == '1'
    assert tab.stop_flag.is_set() and tab._global_cancellation is True
    assert tab.is_running is False and tab._batch_mode_active is False
    assert widget.stop_translation_btn.text() == "⏹ Click again to force stop"
    widget._on_stop_translation_clicked()  # second click inside a second: force
    assert os.environ['TRANSLATION_CANCELLED'] == '1' and os.environ['GRACEFUL_STOP'] == '0'
    assert tab._forced is True and FakeMangaTranslator.is_globally_cancelled()
    _qt_teardown(qapp, tab, widget)


# ===========================================================================
# M: the mobile editor session
# ===========================================================================

def _session(tmp_path, pages, config=None, **kwargs):
    owner = FakeMainGui(json.loads(json.dumps(config or BASE_CONFIG)))
    if FONT_PATH.exists():
        owner.config['manga_font_path'] = str(FONT_PATH)
    logs = []
    session = core.MangaEditorSession(owner, state_file=str(tmp_path / "state" / "image_state.json"),
                                      image_paths=pages, log_callback=lambda message, level: logs.append((level, message)),
                                      default_ocr_prompt='DEFAULT OCR PROMPT', **kwargs)
    session.test_logs = logs
    return session


def test_session_takes_the_tab_settings_from_manga_env(tmp_path):
    config = dict(BASE_CONFIG, manga_skip_inpainting=True, manga_text_color=[1, 2, 3], manga_full_page_context=False,
                  manga_settings={'inpainting': {'method': 'local', 'local_method': 'lama_onnx',
                                                 'lama_onnx_model_path': 'x.json'}})
    session = _session(tmp_path, [], config)
    assert session.skip_inpainting_value is True
    assert (session.text_color_r_value, session.text_color_g_value, session.text_color_b_value) == (1, 2, 3)
    assert session.local_model_type_value == 'lama_onnx' and session.local_model_path_value == ''
    assert session.full_page_context_value is False
    assert session.ocr_prompt  # the tab's prompt load (config value or the shared default)
    assert core._get_inpaint_config(session)['skip'] is True


def test_session_workflow_steps(tmp_path, fakes):
    import multiprocessing
    pages = [make_page(tmp_path / "pages" / "001.png"), make_page(tmp_path / "pages" / "002.png", labels=("konbanwa", "sekai"))]
    session = _session(tmp_path, pages)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(multiprocessing, "get_context", _spawn_tripwire)
        mp.setattr(multiprocessing, "Process", _spawn_tripwire)
        mp.setattr(time, "sleep", lambda seconds: None)
        session.open_page(pages[0])
        regions = session.detect()
        assert [r['bbox'] for r in regions] == [list(b) for b in BOXES]
        assert [b.bubble_type for b in session.boxes] == ['text_bubble', 'free_text']
        recognized = session.recognize()
        assert [(r['region_index'], r['text']) for r in recognized] == [(0, 'konnichiwa'), (1, 'sekai')]
        assert all(b.is_recognized for b in session.boxes)
        result = session.translate()
        assert [t['translation'] for t in result['translated_texts']] == ['Hello there', 'World']
        assert result['rendered_path'] and os.path.isfile(result['rendered_path'])
        cleaned = session.image_state_manager.get_state(pages[0]).get('cleaned_image_path')
        assert cleaned and os.path.isfile(cleaned)
        outputs = session.translate_all()
    assert set(outputs) == set(os.path.abspath(p) for p in pages)
    assert all(path and os.path.isfile(path) for path in outputs.values())
    saved = json.loads((tmp_path / "state" / "image_state.json").read_text(encoding="utf-8"))
    assert {k: v.get('step') for k, v in saved.items()} == {p.replace("\\", "/"): 'translated' for p in map(os.path.abspath, pages)}
    assert not [m for level, m in session.test_logs if level == 'error'], session.test_logs


def test_session_per_box_actions(tmp_path, fakes):
    page = make_page(tmp_path / "pages" / "001.png")
    session = _session(tmp_path, [page])
    session.open_page(page)
    session.detect()
    session.recognize()
    state = lambda: session.image_state_manager.get_state(page)  # noqa: E731

    assert session.set_box_free_text(0, True) is True
    assert state()['detection_regions'][0]['bubble_type'] == 'free_text'
    assert session.set_box_iterations(1, 6) == 6 and state()['inpaint_iterations'] == {'1': 6}
    assert session.set_box_iterations(1, -1) is None and state()['inpaint_iterations'] == {}
    with pytest.raises(ValueError):
        session.set_box_iterations(1, 51)
    assert session.set_box_excluded(0, True) is True

    assert session.edit_box_text(0, ocr_text='konbanwa') is True
    assert session._recognition_data[0]['text'] == 'konbanwa'
    assert [r for r in state()['recognized_texts'] if r.get('region_index') == 0][0]['text'] == 'konbanwa'
    assert session.translate_box(0) == 'Good evening'
    assert state()['translated_texts'][0]['translation'] == 'Good evening'
    assert state().get('rendered_image_path') and os.path.isfile(state()['rendered_image_path'])

    assert session.edit_box_text(1, translation='Earth') is True
    assert state()['translated_texts'][1]['translation'] == 'Earth'
    rendered = session.save_and_update_overlay()
    assert rendered and os.path.isfile(rendered)

    box = session.add_box(10, 200, 60, 40, shape='polygon', polygon=[[10, 200], [70, 210], [40, 240]])
    assert box.region_index == 2 and state()['viewer_rectangles'][2]['polygon']
    assert session.ocr_box(2) == ''  # nothing to read there (crop width not in the fixture map)
    revision = session.output_revision
    assert session.update_box(1, 230, 175, 140, 85) is not None
    assert session.output_revision > revision

    cleaned = session.clean_box(1)
    assert cleaned and os.path.isfile(cleaned)
    session.delete_box(2)
    assert len(session.boxes) == 2
    assert not [m for level, m in session.test_logs if level == 'error'], session.test_logs


def test_session_clean_skips_excluded_boxes(tmp_path, fakes):
    page = make_page(tmp_path / "pages" / "001.png")
    session = _session(tmp_path, [page])
    session.open_page(page)
    session.detect()
    session.set_box_excluded(1, True)
    cleaned = session.clean()
    assert cleaned and os.path.isfile(cleaned)
    source, result = cv2.imread(page), cv2.imread(cleaned)
    x, y, w, h = BOXES[1]
    assert np.array_equal(source[y:y + h, x:x + w], result[y:y + h, x:x + w])
    x, y, w, h = BOXES[0]
    assert not np.array_equal(source[y:y + h, x:x + w], result[y:y + h, x:x + w])


def test_session_reopens_pages_from_state(tmp_path, fakes):
    pages = [make_page(tmp_path / "pages" / "001.png"), make_page(tmp_path / "pages" / "002.png")]
    session = _session(tmp_path, pages)
    session.open_page(pages[0])
    session.detect()
    session.recognize()
    session.translate()
    session.open_page(pages[1])
    assert session.boxes == [] and not session._recognition_data
    session.close()

    reopened = _session(tmp_path, pages)
    snapshot = reopened.open_page(pages[0])
    assert [b['ocr_text'] for b in snapshot['boxes']] == ['konnichiwa', 'sekai']
    assert [b['translation'] for b in snapshot['boxes']] == ['Hello there', 'World']
    assert snapshot['rendered_path'] and os.path.isfile(snapshot['rendered_path'])
    assert all(b['is_recognized'] for b in snapshot['boxes'])


def test_session_stop_graceful_and_force(tmp_path, fakes):
    pages = [make_page(tmp_path / "pages" / f"{i:03d}.png") for i in range(1, 4)]
    session = _session(tmp_path, pages)
    session.open_page(pages[0])
    started = threading.Event()
    release = threading.Event()

    def slow_send(text):
        started.set()
        release.wait(10)

    FakeUnifiedClient.send_hook = slow_send
    worker = threading.Thread(target=session.translate_all, kwargs={'owner': None})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(time, "sleep", lambda seconds: None)
        worker.start()
        assert started.wait(20)
        session.stop()
        assert session.stop_flag.is_set() and os.environ['GRACEFUL_STOP'] == '1'
        release.set()
        worker.join(60)
    assert not worker.is_alive()
    translated = [p for p in pages if (session.image_state_manager.get_state(p) or {}).get('step') == 'translated']
    assert len(translated) < len(pages)
    session.stop(force=True)
    assert os.environ['TRANSLATION_CANCELLED'] == '1' and FakeMangaTranslator.is_globally_cancelled()
    FakeUnifiedClient.send_hook = None
    # the next step resets the stop flags (the desktop click handlers' _reset_cancellation_flags)
    session.detect(pages[1])
    assert not session.stop_flag.is_set() and os.environ['GRACEFUL_STOP'] == '0'


def test_session_ocr_export_import_round_trip(tmp_path, fakes):
    pages = [make_page(tmp_path / "pages" / "001.png"), make_page(tmp_path / "pages" / "002.png")]
    session = _session(tmp_path, pages)
    session.open_page(pages[0])
    session.detect()
    session.recognize()
    session.translate()
    session.edit_box_text(1, translation='Earth')
    exported = session.export_ocr(str(tmp_path / "out" / "session.json"))
    assert exported['pages'] == 1 and exported['translated_regions'] == 2
    document = json.loads(Path(exported['path']).read_text(encoding="utf-8"))
    assert document['format'] == 'glossarion-manga-ocr' and document['workflow'] == 'manual-editor'

    moved = tmp_path / "moved"
    moved.mkdir()
    copies = [shutil.copy(p, str(moved / Path(p).name)) for p in pages]
    other = _session(tmp_path / "other", copies)
    result = other.import_ocr(exported['path'])
    assert result['matched'] == 1 and result['translated_regions'] == 2
    rendered = result['rendered'][os.path.abspath(copies[0])]
    assert os.path.isfile(rendered)
    snapshot = other.open_page(copies[0])
    assert [b['translation'] for b in snapshot['boxes']] == ['Hello there', 'Earth']


def test_session_with_a_headless_owner(tmp_path, fakes, monkeypatch):
    """HeadlessOwner (the mobile job's owner) is the duck-typed main_gui of the editor functions."""
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.json"))
    from headless_owner import HeadlessOwner
    config = json.loads(json.dumps(BASE_CONFIG))
    owner = HeadlessOwner(config, api_key='test-key', model='gpt-test')
    page = make_page(tmp_path / "pages" / "001.png")
    session = core.MangaEditorSession(state_file=str(tmp_path / "state.json"), image_paths=[page],
                                      default_ocr_prompt='DEFAULT OCR PROMPT')
    with pytest.raises(ValueError):
        session.detect(page)
    session.set_owner(owner)
    session.open_page(page)
    session.detect(owner=owner)
    session.recognize(owner=owner)
    result = session.translate(owner=owner)
    assert [t['translation'] for t in result['translated_texts']] == ['Hello there', 'World']
    assert os.path.isfile(result['rendered_path'])


def test_session_renders_with_the_real_renderer(tmp_path, fakes):
    """Save & Update Overlay on mobile through the real PIL renderer (fixed font)."""
    assert FONT_PATH.exists()
    page = make_page(tmp_path / "pages" / "001.png")
    session = _session(tmp_path, [page])
    session.open_page(page)
    session.detect()
    session.recognize()
    session.translate()
    session._manga_translator = _real_manga_translator()
    rendered_path = session.save_and_update_overlay()
    rendered = cv2.imread(rendered_path)
    cleaned = cv2.imread(session.image_state_manager.get_state(page)['cleaned_image_path'])
    assert rendered.shape == cleaned.shape
    assert int(np.count_nonzero(np.any(rendered != cleaned, axis=2))) > 200


# ===========================================================================
# G: google_vision_rest in the editor OCR when the SDK is missing
# ===========================================================================

def _block_vision_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.cloud.vision", None)
    google_cloud = sys.modules.get("google.cloud")
    if google_cloud is not None and hasattr(google_cloud, "vision"):
        monkeypatch.delattr(google_cloud, "vision")


def test_import_google_vision_prefers_the_sdk_then_rest(monkeypatch):
    try:
        from google.cloud import vision as sdk  # noqa: F401
    except ImportError:
        sdk = None
    if sdk is not None:
        assert core._import_google_vision() is sdk
    _block_vision_sdk(monkeypatch)
    import google_vision_rest
    assert core._import_google_vision() is google_vision_rest.vision


def test_editor_google_ocr_uses_rest_without_the_sdk(tmp_path, monkeypatch, fakes):
    import http.server

    class Handler(http.server.BaseHTTPRequestHandler):
        requests = []

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            Handler.requests.append((self.path, json.loads(body)))
            annotations = [{"description": "konnichiwa sekai"}]
            for (x, y, w, h), text in zip(BOXES, ("konnichiwa", "sekai")):
                annotations.append({"description": text, "boundingPoly": {"vertices": [
                    {"x": x + 5, "y": y + 5}, {"x": x + w - 5, "y": y + 5}, {"x": x + w - 5, "y": y + h - 5},
                    {"x": x + 5, "y": y + h - 5}]}})
            payload = json.dumps({"responses": [{"textAnnotations": annotations}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _block_vision_sdk(monkeypatch)
        monkeypatch.setenv("GOOGLE_VISION_REST_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setenv("GOOGLE_VISION_API_KEY", "test-vision-key")
        creds = tmp_path / "creds.json"
        creds.write_text("{}", encoding="utf-8")
        page = make_page(tmp_path / "001.png")
        host = TraceHost(tmp_path, BASE_CONFIG, page)
        texts = core._run_ocr_on_regions(host, page, [{'bbox': list(b), 'confidence': 1.0} for b in BOXES],
                                         {'provider': 'google', 'google_credentials_path': str(creds)})
    finally:
        server.shutdown()
    assert [(t['region_index'], t['text']) for t in texts] == [(0, 'konnichiwa'), (1, 'sekai')]
    assert Handler.requests and Handler.requests[0][0].startswith("/v1/images:annotate")
    assert os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') == str(creds)
