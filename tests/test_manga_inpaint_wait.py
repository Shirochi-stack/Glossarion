"""The manga editor's Translate never renders on the original page while inpainting is enabled.

Owner report (2026-11-06): after switching the Local Model from custom-image-edit to lama_onnx, the
first Translate downloaded LaMa ONNX inside its concurrent inpainting thread, gave up waiting after
30 s ("Inpainting still running, rendering on original image") and left the original Korean under
the English; Skip Inpainter stayed locked meanwhile.

* ``_run_translate_background`` waits for the inpainting thread with every local model (not only the
  custom image edit), with progress lines; only a force stop ends the wait.
* ``_run_inpainting_sync`` returns its pool checkout on the error / force-stop exits too: a failed
  custom image edit request used to stay "in use", which kept Skip Inpainter locked.
* Switching the Local Model refreshes Skip Inpainter's lock at once.

The slow inpainter is faked; ``threading.Thread.join`` timeouts run 100x faster, so a 1 s fake
download stands for a 100 s one (longer than the old 30 s wait).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from types import MethodType, SimpleNamespace

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import local_inpainter  # noqa: E402
import manga_editor_core  # noqa: E402
from manga_translator import MangaTranslator  # noqa: E402


class _FastClockThread(threading.Thread):
    """A thread whose timed joins return 100x sooner (30 s of waiting takes 0.3 s)."""

    def join(self, timeout=None):
        return super().join(None if timeout is None else timeout / 100.0)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(MangaTranslator, "_inpaint_pool", {})
    monkeypatch.delenv("GRACEFUL_STOP", raising=False)
    MangaTranslator.set_global_cancellation(False)
    yield
    MangaTranslator.set_global_cancellation(False)


def _editor(config=None):
    logs = []
    editor = SimpleNamespace(
        main_gui=SimpleNamespace(config=dict(config or {})),
        update_queue=Queue(),
        inpaint_method_value="local",
        skip_inpainting_value=False,
        full_page_context_value=False,
        logs=logs,
    )
    editor._log = lambda message, level="info": logs.append((str(message), level))
    return editor


def _queued(editor, kind):
    items = []
    while not editor.update_queue.empty():
        items.append(editor.update_queue.get_nowait())
    return [payload for name, payload, *_ in items if name == kind]


def _translate_with_slow_inpainter(monkeypatch, tmp_path, editor, inpaint):
    page = tmp_path / "page.png"
    page.write_bytes(b"")
    monkeypatch.setattr(threading, "Thread", _FastClockThread)
    monkeypatch.setattr(manga_editor_core, "_run_inpainting_sync", inpaint)
    monkeypatch.setattr(manga_editor_core, "_translate_individually",
                        lambda self, texts, image_path: [dict(texts[0], translation="Hello")])
    texts = [{"text": "안녕", "bbox": [2, 2, 10, 10], "region_index": 0}]
    manga_editor_core._run_translate_background(editor, texts, str(page))
    return str(page)


def test_translate_waits_for_a_slow_first_inpainter_load(monkeypatch, tmp_path):
    """A first run downloads + loads LaMa ONNX inside the inpainting thread (here 1 s = 100 s under the
    fast clock): the render waits for it and uses the cleaned page."""
    cleaned = tmp_path / "page_cleaned.png"
    cleaned.write_bytes(b"")

    def slow_inpaint(self, image_path, regions):
        time.sleep(1.0)  # "📥 Downloading LaMa ONNX (Carve)..." + load
        return str(cleaned)

    monkeypatch.setattr(manga_editor_core._wait_for_concurrent_inpainting, "__defaults__", (0.2,))
    editor = _editor({"manga_inpaint_method": "local", "manga_local_inpaint_model": "lama_onnx"})
    page = _translate_with_slow_inpainter(monkeypatch, tmp_path, editor, slow_inpaint)

    results = _queued(editor, "translate_results")
    assert len(results) == 1
    assert results[0]["image_path"] == str(cleaned)
    assert results[0]["original_image_path"] == page
    messages = [m for m, _ in editor.logs]
    assert not any("rendering on original image" in m for m in messages), messages
    assert any("waiting for inpainting to finish before rendering" in m for m in messages), messages
    assert any("Inpainting finished, rendering on the cleaned image" in m for m in messages), messages


def test_force_stop_ends_the_wait(monkeypatch, tmp_path):
    """Only a force stop ends the wait (the inpainting thread is still busy): the result is then
    rendered on the original page, said as a stop, not as a normal outcome."""
    release = threading.Event()

    def stuck_inpaint(self, image_path, regions):
        release.wait(10)
        return None

    editor = _editor({"manga_inpaint_method": "local", "manga_local_inpaint_model": "lama_onnx"})

    def translate_then_stop(self, texts, image_path):
        self._global_cancellation = True  # Stop pressed (force) while inpainting still runs
        return [dict(texts[0], translation="Hello")]

    monkeypatch.setattr(manga_editor_core, "_translate_individually", translate_then_stop)
    page = tmp_path / "page.png"
    page.write_bytes(b"")
    monkeypatch.setattr(threading, "Thread", _FastClockThread)
    monkeypatch.setattr(manga_editor_core, "_run_inpainting_sync", stuck_inpaint)
    started = time.monotonic()
    try:
        manga_editor_core._run_translate_background(
            editor, [{"text": "안녕", "bbox": [2, 2, 10, 10], "region_index": 0}], str(page))
    finally:
        release.set()
    assert time.monotonic() - started < 5
    results = _queued(editor, "translate_results")
    assert [r["image_path"] for r in results] == [str(page)]
    assert ("⏹ Stopped before inpainting finished, rendering on original image", "warning") in editor.logs


# ---------------------------------------------------------------------------
# _run_inpainting_sync gives its pool checkout back on every exit
# ---------------------------------------------------------------------------

class _FailingInpainter:
    """A loaded inpainter whose request fails (an image edit endpoint answering 400)."""

    on_load = None

    def __init__(self, enable_worker_process=True, **kwargs):
        self.config = {}
        self.log_callback = None
        self.model_loaded = False

    def load_model(self, method, model_path, force_reload=False):
        self.model_loaded = True
        if _FailingInpainter.on_load:
            _FailingInpainter.on_load()
        return True

    def set_log_callback(self, callback):
        self.log_callback = callback

    def inpaint(self, image, mask, **kwargs):
        raise RuntimeError("Custom image edit endpoint error (400): bad size")


def _checked_out_count():
    return sum(len(rec.get("checked_out") or []) for rec in MangaTranslator._inpaint_pool.values())


@pytest.mark.parametrize("model, exit_kind", [
    ("custom-image-edit", "error"),
    ("custom-image-edit", "force_stop"),
    ("lama_onnx", "error"),
])
def test_inpainting_sync_returns_its_checkout_on_every_exit(monkeypatch, tmp_path, model, exit_kind):
    page = tmp_path / "page.png"
    cv2.imwrite(str(page), np.full((32, 32, 3), 255, dtype=np.uint8))
    model_file = tmp_path / "lama_fp32.onnx"
    model_file.write_bytes(b"onnx")
    config = {
        "manga_inpaint_method": "local",
        "manga_local_inpaint_model": model,
        "custom_image_edit_endpoint": "http://127.0.0.1:9/v1",
        "manga_lama_onnx_model_path": str(model_file),
    }
    editor = _editor(config)
    monkeypatch.setattr(local_inpainter, "LocalInpainter", _FailingInpainter)
    monkeypatch.setattr(_FailingInpainter, "on_load", None)
    if exit_kind == "force_stop":
        # Stop is pressed while the inpainter loads: the next check-point is after the checkout
        monkeypatch.setattr(_FailingInpainter, "on_load",
                            staticmethod(lambda: setattr(editor, "_global_cancellation", True)))

    result = manga_editor_core._run_inpainting_sync(editor, str(page), [{"bbox": [4, 4, 12, 12], "text": "안녕"}])

    assert result is None
    assert sum(len(rec.get("spares") or []) for rec in MangaTranslator._inpaint_pool.values()) == 1
    # Nothing stays "in use": the manga tab's tracker unlocks Skip Inpainter
    assert _checked_out_count() == 0
    assert ("update_pool_tracker", None) in [editor.update_queue.get_nowait()
                                             for _ in range(editor.update_queue.qsize())]


# ---------------------------------------------------------------------------
# The Local Model switch refreshes Skip Inpainter's lock at once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize("new_model", ["lama_onnx", "custom-image-edit"])
def test_local_model_switch_refreshes_skip_inpainter_lock(qapp, new_model):
    from PySide6.QtWidgets import QCheckBox, QLabel, QLineEdit
    from manga_integration import MangaTranslationTab

    tab = SimpleNamespace(
        _model_loading_in_progress=False,
        local_model_type_value="custom-image-edit" if new_model != "custom-image-edit" else "lama_onnx",
        main_gui=SimpleNamespace(config={}),
        model_desc_label=QLabel(),
        local_model_entry=QLineEdit(),
        local_model_status_label=QLabel(),
        skip_inpainting_checkbox=QCheckBox("Skip Inpainter"),
    )
    for name in ("_apply_custom_image_edit_ui_state", "_save_rendering_settings", "_clear_inpainter_preload_failure",
                 "_set_translation_buttons_waiting", "_sync_custom_image_edit_controls"):
        setattr(tab, name, lambda *a, **k: None)
    tab._is_local_inpainting_enabled = lambda: True
    tab._get_inpainter_pool_usage = MethodType(MangaTranslationTab._get_inpainter_pool_usage, tab)
    tab._sync_skip_inpainter_pool_lock = MethodType(MangaTranslationTab._sync_skip_inpainter_pool_lock, tab)

    # A request of the previous model was checked out: Skip Inpainter is locked
    stale = object()
    MangaTranslator._inpaint_pool[("custom-image-edit", "http://x/v1")] = {"spares": [stale], "checked_out": [stale]}
    tab._sync_skip_inpainter_pool_lock()
    assert not tab.skip_inpainting_checkbox.isEnabled()

    # It went back to the pool; the switch shows that at once (no 2 s tracker tick here)
    MangaTranslator._inpaint_pool[("custom-image-edit", "http://x/v1")]["checked_out"].clear()
    MangaTranslationTab._on_local_model_change(tab, new_model)

    assert tab.local_model_type_value == new_model
    assert tab.skip_inpainting_checkbox.isEnabled()
