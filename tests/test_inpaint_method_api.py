"""Manga Inpainting: Local and API are separate methods (owner, 2026-11-07).

The radios are "Replicate (Bad)", "Local (Light-weight)", "API (Recommended)" and "Hybrid (Trash)".
API is the custom image edit inpainter (``manga_inpaint_method`` 'api', local model
'custom-image-edit'); the Local dropdown lists the local models only.

* ``manga_models.migrate_inpaint_method``: an old 'local' + 'custom-image-edit' config moves to 'api'
  (top-level and nested copies); a real local model, Replicate and Hybrid stay.
* The run path treats 'api' as the custom image edit inpainter (and an old unsaved 'local' +
  'custom-image-edit' config the same way): the editor's Clean/Translate, the translator's live model.
* The desktop tab (offscreen Qt): the four radios, the migration on open, and which controls each shows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC = TESTS_DIR.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import manga_models  # noqa: E402

OLD_API = {"manga_inpaint_method": "local", "manga_local_inpaint_model": "custom-image-edit",
           "manga_settings": {"inpainting": {"method": "local", "local_method": "custom-image-edit"}}}


# ---------------------------------------------------------------------------
# migration
# ---------------------------------------------------------------------------

def test_old_local_custom_image_edit_config_moves_to_api():
    config = json.loads(json.dumps(OLD_API))
    assert manga_models.migrate_inpaint_method(config) is True
    assert config["manga_inpaint_method"] == "api"
    assert config["manga_local_inpaint_model"] == "custom-image-edit"
    assert config["manga_settings"]["inpainting"] == {"method": "api", "local_method": "custom-image-edit"}
    assert manga_models.migrate_inpaint_method(config) is False  # idempotent


@pytest.mark.parametrize("config", [
    {"manga_inpaint_method": "local", "manga_local_inpaint_model": "lama_onnx"},
    {"manga_inpaint_method": "local", "manga_local_inpaint_model": "anime_onnx",
     "manga_settings": {"inpainting": {"method": "local", "local_method": "custom-image-edit"}}},
    {"manga_inpaint_method": "cloud", "manga_local_inpaint_model": "custom-image-edit"},
    {"manga_inpaint_method": "hybrid", "manga_local_inpaint_model": "custom-image-edit"},
    {},
])
def test_real_local_models_replicate_and_hybrid_stay(config):
    before = json.loads(json.dumps(config))
    assert manga_models.migrate_inpaint_method(config) is False
    assert config == before


def test_migration_reads_the_nested_copy_and_old_model_names():
    nested = {"manga_settings": {"inpainting": {"method": "local", "local_method": "custom-image-edit"}}}
    assert manga_models.migrate_inpaint_method(nested) is True
    assert nested["manga_inpaint_method"] == "api" and nested["manga_settings"]["inpainting"]["method"] == "api"
    qwen = {"manga_inpaint_method": "local", "manga_local_inpaint_model": "qwen_image_edit"}
    assert manga_models.migrate_inpaint_method(qwen) is True
    assert qwen == {"manga_inpaint_method": "api", "manga_local_inpaint_model": "custom-image-edit"}
    # an 'api' config always carries the custom-image-edit model
    stale = {"manga_inpaint_method": "api", "manga_local_inpaint_model": "lama_onnx"}
    assert manga_models.migrate_inpaint_method(stale) is True
    assert stale["manga_local_inpaint_model"] == "custom-image-edit"


def test_normalize_and_effective_model():
    assert manga_models.normalize_inpaint_method("local", "custom-image-edit") == "api"
    assert manga_models.normalize_inpaint_method("api", "lama_onnx") == "api"
    assert manga_models.normalize_inpaint_method("local", "lama_onnx") == "local"
    assert manga_models.normalize_inpaint_method("cloud", "custom-image-edit") == "cloud"
    assert manga_models.normalize_inpaint_method(None, None) == "local"
    assert manga_models.effective_inpaint_model("api", "lama_onnx") == "custom-image-edit"
    assert manga_models.effective_inpaint_model("local", "lama_onnx") == "lama_onnx"
    assert manga_models.effective_inpaint_model("local", "") == "anime_onnx"
    # the API method downloads no local ONNX model
    assert manga_models.required_models({"manga_inpaint_method": "api",
                                         "manga_settings": {"ocr": {"bubble_detection_enabled": False}}}) == []


# ---------------------------------------------------------------------------
# the run path treats 'api' as the custom image edit inpainter
# ---------------------------------------------------------------------------

class _RecordingInpainter:
    loads: list = []

    def __init__(self, enable_worker_process=True, **kwargs):
        self.config = {}
        self.model_loaded = False

    def load_model(self, method, model_path, force_reload=False):
        _RecordingInpainter.loads.append((method, model_path))
        self.model_loaded = True
        return True

    def set_log_callback(self, callback):
        pass

    def inpaint(self, image, mask, **kwargs):
        raise RuntimeError("stop here")


@pytest.mark.parametrize("method, model, expected", [
    ("api", "custom-image-edit", "custom-image-edit"),
    ("api", "lama_onnx", "custom-image-edit"),          # a stale local model never runs under API
    ("local", "custom-image-edit", "custom-image-edit"),  # old config, not saved yet
    ("local", "lama_onnx", "lama_onnx"),
])
def test_editor_clean_uses_the_custom_image_edit_inpainter_for_api(monkeypatch, tmp_path, method, model, expected):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    import local_inpainter
    import manga_editor_core
    from manga_translator import MangaTranslator

    monkeypatch.setattr(MangaTranslator, "_inpaint_pool", {})
    page = tmp_path / "page.png"
    cv2.imwrite(str(page), np.full((32, 32, 3), 255, dtype=np.uint8))
    model_file = tmp_path / "lama_fp32.onnx"
    model_file.write_bytes(b"onnx")
    editor = SimpleNamespace(
        main_gui=SimpleNamespace(config={
            "manga_inpaint_method": method, "manga_local_inpaint_model": model,
            "custom_image_edit_endpoint": "http://127.0.0.1:9/v1", "manga_lama_onnx_model_path": str(model_file)}),
        update_queue=Queue(), inpaint_method_value=method, skip_inpainting_value=False,
        full_page_context_value=False)
    editor._log = lambda message, level="info": None
    monkeypatch.setattr(local_inpainter, "LocalInpainter", _RecordingInpainter)
    monkeypatch.setattr(_RecordingInpainter, "loads", [])

    manga_editor_core._run_inpainting_sync(editor, str(page), [{"bbox": [4, 4, 12, 12], "text": "x"}])

    assert [m for m, _ in _RecordingInpainter.loads] == [expected]


def test_translator_live_model_is_custom_image_edit_under_api():
    from manga_translator import MangaTranslator

    combo = SimpleNamespace(currentText=lambda: "anime_onnx")  # the Local dropdown keeps its own model
    tab = SimpleNamespace(inpaint_method_value="api", local_model_combo=combo, local_model_type_value="anime_onnx")
    translator = SimpleNamespace(main_gui=SimpleNamespace(manga_translator=tab, config={}), manga_settings={})
    assert MangaTranslator._get_live_local_inpaint_method(translator) == "custom-image-edit"
    tab.inpaint_method_value = "local"
    assert MangaTranslator._get_live_local_inpaint_method(translator) == "anime_onnx"
    headless = SimpleNamespace(main_gui=SimpleNamespace(config={"manga_inpaint_method": "api",
                                                                "manga_local_inpaint_model": "lama_onnx"}),
                               manga_settings={})
    assert MangaTranslator._get_live_local_inpaint_method(headless) == "custom-image-edit"


def test_tab_counts_api_as_local_pipeline_inpainting():
    pytest.importorskip("PySide6")
    import manga_integration

    def enabled(method):
        tab = SimpleNamespace(main_gui=SimpleNamespace(config={
            "manga_settings": {"inpainting": {"method": method}}}))
        return manga_integration.MangaTranslationTab._is_local_inpainting_enabled(tab)

    assert enabled("api") and enabled("local") and not enabled("cloud") and not enabled("hybrid")


# ---------------------------------------------------------------------------
# the desktop tab (offscreen Qt): four radios, migration on open, the controls each shows
# ---------------------------------------------------------------------------

_TAB_SCRIPT = textwrap.dedent(r"""
    import json, os, sys, types
    from PySide6.QtWidgets import QApplication, QDialog, QScrollArea, QWidget
    app = QApplication.instance() or QApplication([])
    import ImageRenderer, manga_integration
    ImageRenderer._preload_shared_bubble_detector = lambda *_a, **_k: None
    ImageRenderer._preload_shared_inpainter = lambda *_a, **_k: True
    class StateManager:
        def __init__(self, path): self.states = {}; self._timer_lock = None
        def get_state(self, p): return self.states.get(p, {})
        def set_state(self, p, s, save=True): self.states[p] = s
        def update_state(self, p, u, save=True): self.states.setdefault(p, {}).update(u)
        def flush(self): pass
        def flush_async(self): pass
    manga_integration.ImageStateManager = StateManager
    from headless_owner import HeadlessOwner
    results = {}
    WIDGETS = ("cloud_inpaint_frame", "local_inpaint_frame", "local_model_frame", "custom_image_edit_controls_frame",
               "custom_image_edit_output_frame", "batch_image_requests_frame", "model_info_btn",
               "disable_inpaint_performance_mode_frame", "load_model_btn")
    for name, config in json.loads(sys.argv[1]).items():
        owner = HeadlessOwner({'model': 'gpt-4o-mini', 'api_key': 'sk', **config},
                              host=types.SimpleNamespace(log=lambda *a, **k: None))
        owner.save_config = lambda show_message=True: True
        tab = manga_integration.MangaTranslationTab(QWidget(), owner, QDialog(), QScrollArea())
        for _ in range(5):
            app.processEvents()
        group = tab.inpaint_method_frame.parentWidget()
        radios = (tab.cloud_radio, tab.local_radio, tab.api_radio, tab.hybrid_radio)

        def state():
            nested = owner.config.get('manga_settings', {}).get('inpainting', {})
            return {"checked": [r.text() for r in radios if r.isChecked()],
                    "method": tab.inpaint_method_value, "model": tab.local_model_type_value,
                    "config": [owner.config.get('manga_inpaint_method'), owner.config.get('manga_local_inpaint_model'),
                               nested.get('method'), nested.get('local_method')],
                    "visible": sorted(w for w in WIDGETS if getattr(tab, w).isVisibleTo(group))}

        entry = {"labels": [r.text() for r in radios],
                 "layout": [r.parentWidget().layout().indexOf(r) for r in radios],
                 "combo": [tab.local_model_combo.itemText(i) for i in range(tab.local_model_combo.count())],
                 "open": state()}
        for radio in ("local_radio", "api_radio", "hybrid_radio", "cloud_radio"):
            getattr(tab, radio).setChecked(True)
            for _ in range(3):
                app.processEvents()
            entry[radio] = state()
        results[name] = entry
    print("RESULT " + json.dumps(results), flush=True)
    os._exit(0)  # os._exit skips the stdio flush
""")


@pytest.fixture(scope="module")
def tab_states(tmp_path_factory):
    pytest.importorskip("PySide6")
    tmp_path = tmp_path_factory.mktemp("inpaint_tab")
    configs = {"old_api": OLD_API,
               "old_lama": {"manga_inpaint_method": "local", "manga_local_inpaint_model": "lama_onnx"}}
    env = dict(os.environ, PYTHONIOENCODING="utf-8", QT_QPA_PLATFORM="offscreen", HOME=str(tmp_path),
               USERPROFILE=str(tmp_path), GLOSSARION_LIBRARY_DIR=str(tmp_path / "library"),
               CONFIG_FILE=str(tmp_path / "config.json"), YOLO_CONFIG_DIR=str(tmp_path / "yolo"),
               GLOSSARION_HTTP_LOG="0")
    env["PYTHONPATH"] = os.pathsep.join([str(SRC)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
    proc = subprocess.run([sys.executable, "-c", _TAB_SCRIPT, json.dumps(configs)], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, proc.stdout[-3000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT "):])


API_CONTROLS = ["batch_image_requests_frame", "custom_image_edit_controls_frame", "custom_image_edit_output_frame",
                "local_inpaint_frame"]
LOCAL_CONTROLS = ["disable_inpaint_performance_mode_frame", "load_model_btn", "local_inpaint_frame",
                  "local_model_frame", "model_info_btn"]


def test_four_radios_and_a_local_only_dropdown(tab_states):
    entry = tab_states["old_api"]
    assert entry["labels"] == ["Replicate (Bad)", "Local (Light-weight)", "API (Recommended)", "Hybrid (Trash)"]
    assert entry["layout"] == [0, 1, 2, 3]
    assert "custom-image-edit" not in entry["combo"]
    assert {"aot", "aot_onnx", "lama", "lama_onnx", "anime", "anime_onnx", "mat"} <= set(entry["combo"])


def test_old_custom_image_edit_config_opens_on_api(tab_states):
    opened = tab_states["old_api"]["open"]
    assert opened["checked"] == ["API (Recommended)"]
    assert (opened["method"], opened["model"]) == ("api", "custom-image-edit")
    assert opened["config"] == ["api", "custom-image-edit", "api", "custom-image-edit"]
    assert opened["visible"] == API_CONTROLS


def test_old_real_local_model_opens_on_local(tab_states):
    opened = tab_states["old_lama"]["open"]
    assert opened["checked"] == ["Local (Light-weight)"]
    assert (opened["method"], opened["model"]) == ("local", "lama_onnx")
    assert opened["config"][:2] == ["local", "lama_onnx"]
    assert opened["visible"] == LOCAL_CONTROLS


def test_each_radio_shows_its_own_controls_and_saves_its_method(tab_states):
    entry = tab_states["old_api"]
    local = entry["local_radio"]
    assert (local["method"], local["model"]) == ("local", "anime_onnx")  # the dropdown's model, never custom-image-edit
    assert local["config"] == ["local", "anime_onnx", "local", "anime_onnx"]
    assert local["visible"] == LOCAL_CONTROLS
    api = entry["api_radio"]
    assert api["config"] == ["api", "custom-image-edit", "api", "custom-image-edit"]
    assert api["visible"] == API_CONTROLS
    hybrid = entry["hybrid_radio"]
    assert hybrid["config"][0] == "hybrid" and hybrid["model"] == "anime_onnx"
    assert hybrid["visible"] == sorted(["cloud_inpaint_frame"] + LOCAL_CONTROLS)
    cloud = entry["cloud_radio"]
    assert cloud["config"][0] == "cloud" and cloud["visible"] == ["cloud_inpaint_frame"]
    lama_api = tab_states["old_lama"]["api_radio"]
    assert lama_api["config"][:2] == ["api", "custom-image-edit"] and lama_api["visible"] == API_CONTROLS
    assert tab_states["old_lama"]["local_radio"]["config"][:2] == ["local", "lama_onnx"]  # dropdown kept its model
