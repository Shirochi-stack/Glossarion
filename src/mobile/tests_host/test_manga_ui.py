"""Host tests for Tools › Manga (U8): the services binding layer, the MANGA / MANGA_STEP job
adapters, the Files / Settings / Editor tabs, the box sheet, model rows, the Open-with action
and the chat's "Translate as manga" entry points.

* ``services.manga`` runs over the real shared cores where they are cheap and side-effect free
  (``manga_files_core`` / ``manga_env`` mixins for the Files model, ``manga_settings_defaults``,
  ``manga_models`` statuses, ``settings_schema`` value rules) on fixture folders in pytest tmp
  dirs; the heavy ones (``manga_runner``, ``manga_editor_core`` sessions) are fakes that follow
  their contracts (``HeadlessMangaRunner(owner, host=, files=, ...)``.run() / request_stop,
  ``MangaEditorSession`` methods), so no model, network or translator ever runs here.
* Screens are built in the in-memory Flet session of ``test_bootstrap``; dialogs are scripted
  through ``ctx.extras["answers"]``.

Real data is never touched: OUTPUT_DIRECTORY, HOME, USERPROFILE, GLOSSARION_LIBRARY_DIR,
GLOSSARION_DATA_DIR and the model cache variables point at the test's tmp dir, and HTTP logging
is off.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_manga_ui.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import struct
import sys
import threading
import types
import zipfile
import zlib
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
from glossarion_mobile.job_kinds import manga as manga_kind  # noqa: E402
from glossarion_mobile.services import manga as svc  # noqa: E402
from glossarion_mobile.services.jobs import JobError, JobSnapshot, JobSpec, JobState  # noqa: E402


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")
needs_cores = pytest.mark.skipif(not (_has("manga_files_core") and _has("manga_env")),
                                 reason="the shared manga cores are not importable")

MANGA_KINDS = {"manga": "manga", "manga_step": "manga"}


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


def _tb():
    return _load("_glossarion_u8_tb_helpers", "test_bootstrap.py")


def _jobs_helpers():
    return _load("_glossarion_u8_jobs_helpers", "test_jobs.py")


def write_png(path, width: int = 40, height: int = 60) -> str:
    """A small white RGB PNG without PIL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return str(path)


@pytest.fixture(autouse=True)
def iso(tmp_path, monkeypatch):
    """Every path the manga code may write to lives in tmp_path (every test: the moved desktop code
    falls back to the app folder, i.e. src/, for OCR exports and glossary backups)."""
    paths = {name: tmp_path / name for name in ("Output", "home", "Library", "data", "models")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(paths["Output"]))
    monkeypatch.setenv("HOME", str(paths["home"]))
    monkeypatch.setenv("USERPROFILE", str(paths["home"]))
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(paths["Library"]))
    monkeypatch.setenv("GLOSSARION_DATA_DIR", str(paths["data"]))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.json"))  # never src/config.json
    for name, sub in (("BUBBLE_CACHE_DIR", "detector"), ("MODEL_CACHE_DIR", "inpainting"), ("ONNX_CACHE_DIR", "onnx")):
        monkeypatch.setenv(name, str(paths["models"] / sub))
    monkeypatch.delenv("GLOSSARION_MOBILE", raising=False)
    return paths


@pytest.fixture
def manga_kinds(monkeypatch):
    """Register the U8 kinds (Integrate adds them to ``KIND_MODULES`` / ``JobKind``)."""
    for kind, module in MANGA_KINDS.items():
        monkeypatch.setitem(job_kinds.KIND_MODULES, kind, module)
        job_kinds._CACHE.pop(kind, None)
    yield
    for kind in MANGA_KINDS:
        job_kinds._CACHE.pop(kind, None)


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def make_series(root: Path) -> dict:
    """Inbox/Series/{ch1,ch2}/{1,2,10}.png + a 3-page CBZ + a ZIP of a folder."""
    series = root / "Inbox" / "Series"
    for chapter in ("ch1", "ch2"):
        for n in (1, 2, 10):
            write_png(series / chapter / f"{n}.png")
    cbz = root / "Inbox" / "vol1.cbz"
    with zipfile.ZipFile(cbz, "w") as zf:
        for n in (3, 1, 2):
            zf.write(write_png(root / "tmp" / f"p{n}.png"), f"page{n}.png")
    zipped = root / "Inbox" / "folderzip.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        for n in (1, 2):
            zf.write(write_png(root / "tmp" / f"z{n}.png"), f"sub/z{n}.png")
    return {"series": str(series), "cbz": str(cbz), "zip": str(zipped)}


class FakeCtx:
    """What a job adapter gets (JobContext surface + the private stop mode the bridge reads)."""

    def __init__(self, owner=None, *, params=None, inputs=(), config=None) -> None:
        self.owner = owner if owner is not None else types.SimpleNamespace(config={})
        self.params = dict(params or {})
        self.inputs = tuple(inputs)
        self.config = dict(config or {})
        self.logs: list = []
        self.phases: list = []
        self.outputs: list = []
        self.results: dict = {}
        self.events: list = []
        self.output_dir = None
        self.stop = False
        self._job = types.SimpleNamespace(stop_mode=None)
        self.host = types.SimpleNamespace(emit=lambda kind, **data: self.events.append((kind, data)),
                                          log=self.log, is_graceful_stop=lambda: self._job.stop_mode == "graceful")

    def log(self, text, *args, **kwargs):
        self.logs.append(str(text))

    def stop_requested(self):
        return self.stop

    def request(self, mode):
        self.stop = True
        self._job.stop_mode = mode

    def phase(self, label):
        self.phases.append(label)

    def set_output_dir(self, path):
        self.output_dir = path

    def set_output_dirs(self, mapping):
        pass

    def add_outputs(self, paths):
        self.outputs.extend(paths)

    def set_result(self, **data):
        self.results.update(data)


# ==========================================================================
# services.manga: availability rules, provider / inpainting status, settings defaults
# ==========================================================================


@pytest.mark.skipif(not _has("settings_schema"), reason="settings_schema not importable")
def test_value_rules_come_from_the_schema_and_never_apply_on_desktop():
    for value in ("manga-ocr", "Qwen2-VL", "easyocr", "doctr"):
        assert svc.chip_text(svc.value_reason("manga_ocr_provider", value, mobile=True)) == "Needs PyTorch"
        assert svc.value_reason("manga_ocr_provider", value, mobile=False) is None
    assert svc.value_reason("manga_ocr_provider", "paddleocr", mobile=True).startswith("Needs PaddlePaddle")
    for value in ("custom-api", "google", "azure", "azure-document-intelligence", "rapidocr"):
        assert svc.value_reason("manga_ocr_provider", value, mobile=True) is None
    assert svc.value_reason("manga_local_inpaint_model", "aot", mobile=True)
    assert svc.value_reason("manga_local_inpaint_model", "ollama", mobile=True).startswith("Not functional")
    assert svc.value_reason("manga_local_inpaint_model", "anime_onnx", mobile=True) is None
    assert svc.value_reason("manga_inpaint_method", "hybrid", mobile=True)
    assert svc.value_reason("manga_settings.ocr.detector_type", "yolo", mobile=True)
    assert svc.value_reason("manga_settings.ocr.detector_type", "rtdetr_onnx", mobile=True) is None
    rows = {r.value: r for r in svc.local_model_rows(mobile=True)}
    assert [v for v, r in rows.items() if not r.disabled] == ["aot_onnx", "lama_onnx", "anime_onnx",
                                                              "custom-image-edit"]
    assert "kept for the desktop app" in rows["mat"].detail
    assert {r.value for r in svc.detector_rows(mobile=True) if r.disabled} == {"rtdetr", "yolo", "custom"}
    assert [r.value for r in svc.inpaint_method_rows(mobile=True) if r.disabled] == ["hybrid"]
    assert svc.chip_text("x" * 40).endswith("…") and len(svc.chip_text("x" * 40)) <= 32


def test_ocr_provider_rows_follow_the_desktop_status_rules(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.cloud.vision", None)  # no SDK: the REST fallback
    monkeypatch.setitem(sys.modules, "google_vision_rest", _module("google_vision_rest", vision=object()))
    monkeypatch.setitem(sys.modules, "azure.ai.formrecognizer", None)
    monkeypatch.setitem(sys.modules, "azure_document_intelligence_rest", _module("azure_document_intelligence_rest"))
    monkeypatch.setitem(sys.modules, "azure.ai.vision.imageanalysis", _module("azure.ai.vision.imageanalysis"))
    monkeypatch.setitem(sys.modules, "pyclipper", None)  # RapidOCR without its geometry wheels
    rows = {r.value: r for r in svc.ocr_provider_rows({}, mobile=True)}
    assert [v for v, _label in svc.OCR_PROVIDERS] == list(rows)
    assert rows["custom-api"].status == "ready"
    assert rows["google"].status == "needs_key" and "REST" in rows["google"].detail
    assert rows["azure"].status == "needs_key"
    assert rows["azure-document-intelligence"].detail == "Key & Endpoint needed"
    assert rows["rapidocr"].status == "unavailable" and "pyclipper" in rows["rapidocr"].reason
    assert all(rows[v].disabled for v in ("manga-ocr", "Qwen2-VL", "easyocr", "paddleocr", "doctr"))
    creds = tmp_path / "sa.json"
    creds.write_text(json.dumps({"type": "service_account", "project_id": "p", "private_key": "k",
                                 "client_email": "e@x"}), encoding="utf-8")
    config = {"google_vision_credentials": str(creds), "azure_vision_key": "k",
              "azure_document_intelligence_key": "k", "manga_settings": {"ocr": {"bubble_detection_enabled": False}}}
    rows = {r.value: r for r in svc.ocr_provider_rows(config, mobile=True)}
    assert rows["google"].status == "ready" and rows["google"].detail == "Google Vision REST"
    assert rows["azure"].status == "ready"
    assert rows["azure-document-intelligence"].detail == "Endpoint needed"
    assert rows["custom-api"].detail == "Enable AI bubble detection for best results"
    creds.write_text(json.dumps({"type": "authorized_user"}), encoding="utf-8")
    assert {r.value: r for r in svc.ocr_provider_rows(config, mobile=True)}["google"].status == "needs_key"
    monkeypatch.setitem(sys.modules, "google_vision_rest", None)
    assert {r.value: r for r in svc.ocr_provider_rows({}, mobile=True)}["google"].status == "unavailable"


def test_inpaint_status_and_choice_writes(monkeypatch):
    statuses = {"aot_onnx": svc.ModelEntry("aot_onnx", "AOT ONNX", status="missing")}
    model_status = statuses.get
    assert svc.inpaint_status({"manga_skip_inpainting": True}).chip == "Off"
    cloud = {"manga_inpaint_method": "cloud"}
    assert svc.inpaint_status(cloud).chip == "Needs key"
    assert svc.inpaint_status(dict(cloud, replicate_api_key="r8")).chip == "Ready"
    edit = {"manga_inpaint_method": "local", "manga_local_inpaint_model": "custom-image-edit"}
    assert svc.inpaint_status(edit).chip == "Needs key"
    assert svc.inpaint_status(dict(edit, custom_image_edit_endpoint="https://x/v1")).chip == "Ready"
    local = {"manga_inpaint_method": "local", "manga_local_inpaint_model": "aot_onnx"}
    assert svc.inpaint_status(local, model_status=model_status).chip == "Not downloaded"
    statuses["aot_onnx"] = svc.ModelEntry("aot_onnx", "AOT ONNX", status="downloading", progress=0.42)
    assert svc.inpaint_status(local, model_status=model_status).chip == "Downloading 42%"
    statuses["aot_onnx"] = svc.ModelEntry("aot_onnx", "AOT ONNX", status="loaded")
    assert svc.inpaint_status(local, model_status=model_status).chip == "Preloaded"
    if _has("settings_schema"):
        assert svc.inpaint_status({"manga_inpaint_method": "hybrid"}, mobile=True).status == "unavailable"
    updates = svc.inpaint_choice_updates("local", "aot_onnx")
    assert updates == {"manga_skip_inpainting": False, "manga_inpaint_method": "local",
                       ("manga_settings", "inpainting", "method"): "local", "manga_local_inpaint_model": "aot_onnx",
                       ("manga_settings", "inpainting", "local_method"): "aot_onnx"}
    assert svc.inpaint_choice_updates("skip") == {"manga_skip_inpainting": True}
    # desktop migration of the old local model name
    assert svc.current_inpaint_choice({"manga_local_inpaint_model": "qwen_image_edit"})[1] == "custom-image-edit"


@pytest.mark.skipif(not (_has("manga_settings_defaults") and _has("manga_models")), reason="manga cores missing")
def test_effective_settings_use_the_canonical_and_phone_defaults(iso, monkeypatch):
    assert svc.effective_setting({}, ("manga_settings", "ocr", "detector_type")) == "rtdetr_onnx"
    assert svc.effective_setting({}, ("manga_settings", "ocr", "rtdetr_onnx_variant")) == "detector.onnx"
    assert svc.effective_setting({}, "manga_bg_opacity") == 0  # the tab's _load_rendering_settings default
    stored = {"manga_settings": {"ocr": {"rtdetr_onnx_variant": "detector_int8.onnx"}}, "manga_bg_opacity": 90}
    assert svc.effective_setting(stored, ("manga_settings", "ocr", "rtdetr_onnx_variant")) == "detector_int8.onnx"
    assert svc.effective_setting(stored, "manga_bg_opacity") == 90
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    assert svc.effective_setting({}, ("manga_settings", "ocr", "rtdetr_onnx_variant")) == "detector-v4-s_int8.onnx"
    assert svc.effective_setting({}, "manga_local_inpaint_model") == "aot_onnx"
    assert svc.current_inpaint_choice({}) == ("local", "aot_onnx")
    assert svc.ModelManager().detector_id({}) == "rtdetr_v4_s_int8"


# ==========================================================================
# services.manga: the Files model over the moved MangaFilesMixin methods
# ==========================================================================


@needs_cores
def test_file_list_runs_the_moved_desktop_methods(iso, tmp_path):
    fixture = make_series(tmp_path)
    saved: dict = {}
    files = svc.MangaFileList({}, save=saved.update, temp_root=str(iso["data"] / "manga" / "cbz"),
                              config_source=lambda: dict(saved))
    assert files.available
    assert files.add_paths([fixture["series"]]) == 6
    # _apply_manga_file_sort's default natural order (by file name across the folders)
    assert [os.path.basename(p) for p in files.files] == ["1.png", "1.png", "2.png", "2.png", "10.png", "10.png"]
    assert [(g.name, len(g.files)) for g in files.groups()] == [("Series", 6)]
    files.set_split_first_level(True)
    assert saved["manga_split_first_level_subfolders"] is True
    assert [(g.name, len(g.files)) for g in files.groups()] == [("ch1", 3), ("ch2", 3)]
    assert files.add_paths([fixture["cbz"]]) == 3 and fixture["cbz"] in files.cbz_jobs
    assert files.add_paths([fixture["zip"]]) == 2  # a ZIP is the folder it contains
    # extracted under <folders>/<archive id>/<name>: the folder keeps the archive's name
    zip_root = iso["data"] / "manga" / "folders" / svc._archive_key(fixture["zip"]) / "folderzip"
    assert os.path.isdir(zip_root / "sub") and not os.path.exists(str(zip_root) + ".part")
    assert len(files.files) == 11
    files.set_range("2-4")
    assert files.parse_range() == ({2, 3, 4}, None) and files.range_status() == "3 in range · 3 of 11 will run"
    third = files.files[2]
    assert files.toggle_skip(third) is True and files.is_skipped(third)
    assert [os.path.basename(p) for p in files.run_files()[0]] == [os.path.basename(files.files[1]),
                                                                   os.path.basename(files.files[3])]
    assert saved["manga_skipped_processing_files"] == [files.skip_key(third)]
    assert files.range_skipped(files.files[0]) and not files.range_skipped(files.files[1])
    files.set_range("9-2")
    assert files.run_files()[1] and "start is after end" in files.range_status()
    files.set_range("")
    files.sort("name", reverse=True)
    assert os.path.basename(files.files[0]) == "z2.png"
    first = files.files[0]
    assert files.move(0, 3) and files.files[3] == first and files.host._manga_file_sort is None
    assert saved["manga_selected_files"] == files.files
    assert set(saved["manga_selected_folder_roots"]) == {fixture["series"], str(zip_root)}
    assert files.remove([first]) == 1 and first not in saved["manga_selected_files"]
    reloaded = svc.MangaFileList({}, temp_root=str(iso["data"] / "manga" / "cbz"))
    assert reloaded.load(saved) == len(files.files) - 0 and reloaded.is_skipped(third)
    assert files.ocr_dir() == os.path.join(str(iso["Output"]), "OCR Text")
    assert files.ocr_export_path().startswith(files.ocr_dir())
    out = files.output_path_for(files.files[-1])
    assert out.endswith("_translated" + os.sep + os.path.basename(files.files[-1])) and not os.path.isdir(
        os.path.dirname(out))  # the lookup leaves no empty folder behind
    assert any("Sorted" in line for line, _level in files.host.logs)
    files.clear()
    assert files.files == [] and saved["manga_selected_files"] == []


@needs_cores
def test_batch_spec_hands_the_runner_the_whole_visible_list(iso, tmp_path):
    fixture = make_series(tmp_path)
    files = svc.MangaFileList({}, temp_root=str(iso["data"] / "cbz"))
    files.add_paths([fixture["series"], fixture["cbz"]])
    files.set_range("1-4")
    files.toggle_skip(files.files[0])
    spec = svc.batch_spec(files, output_root=str(iso["Output"]), editor_session="es1")
    params = spec.params
    assert spec.kind == "manga" and not spec.resumable
    assert params["files"] == files.files and params["image_range"] == "1-4"
    assert params["run_files"] == files.run_files()[0] == list(spec.inputs) and len(spec.inputs) == 3
    assert params["skipped"] == [files.skip_key(files.files[0])] and params["editor_session"] == "es1"
    assert params["imported_ocr"] == ""
    assert svc.batch_spec(files, imported_ocr="/x/ocr.json").params["imported_ocr"] == "/x/ocr.json"
    assert params["cbz_jobs"] and params["cbz_image_to_job"] and params["glossary_only"] is False
    json.dumps(spec.to_dict())  # persisted with the job
    assert svc.batch_spec(files, glossary_only=True).title.startswith("Glossary · ")
    files.set_range("40-50")
    with pytest.raises(ValueError, match="No images"):
        svc.batch_spec(files)
    files.set_range("x")
    with pytest.raises(ValueError, match="Invalid image range"):
        svc.batch_spec(files)


def test_step_spec_and_display_copy(tmp_path):
    image = write_png(tmp_path / "p" / "page.png")
    spec = svc.step_spec("translate", "es1", image)
    assert spec.kind == "manga_step" and spec.params == {"step": "translate", "session": "es1",
                                                         "image": os.path.abspath(image)}
    assert spec.title == "Translate · page.png" and not spec.resumable
    with pytest.raises(ValueError):
        svc.step_spec("ocr_box", "es1", image)
    box = svc.step_spec("ocr_box", "es1", image, index=2)
    assert box.params["index"] == 2 and box.title == "OCR this text · box 3 · page.png"
    every = svc.step_spec("translate_all", "es1", image, images=[image, image])
    assert every.inputs == (os.path.abspath(image),) * 2 and every.title == "Translate all · 2 pages"
    with pytest.raises(ValueError):
        svc.step_spec("nope", "es1", image)
    cache = tmp_path / "view"
    first = svc.display_copy(image, 1, str(cache))
    assert os.path.isfile(first) and "_v1_" in os.path.basename(first)
    second = svc.display_copy(image, 2, str(cache))
    assert second != first and not os.path.exists(first) and os.listdir(cache) == [os.path.basename(second)]
    assert svc.display_copy("", 3, str(cache)) == ""


# ==========================================================================
# services.manga: model manager over manga_models
# ==========================================================================


class FakeModels:
    KIND_DETECTOR = "detector"
    MOBILE_DETECTOR_KEY = "det"

    def __init__(self):
        self.calls: list = []
        self.installed: set = set()
        self.partial: dict = {}
        self._loaded: list = []
        self.active: dict = {}
        self.fail = None
        self.specs_list = [
            types.SimpleNamespace(key="det", kind="detector", title="Detector", size=11 * 1024 * 1024,
                                  selector="det.onnx"),
            types.SimpleNamespace(key="aot_onnx", kind="inpaint", title="AOT ONNX", size=22 * 1024 * 1024,
                                  selector="aot_onnx"),
        ]

    def specs(self, kind=None):
        return [s for s in self.specs_list if kind is None or s.kind == kind]

    def get_spec(self, key):
        for spec in self.specs_list:
            if spec.key == key:
                return spec
        raise KeyError(key)

    def status(self, key):
        spec = self.get_spec(key)
        return types.SimpleNamespace(path=f"/m/{key}", installed=key in self.installed,
                                     partial_bytes=self.partial.get(key, 0), downloading=self.active.get(key))

    def download(self, key, progress=None, cancel=None):
        self.calls.append(("download", key))
        report = types.SimpleNamespace(fraction=0.5, phase="download")
        self.active[key] = report
        progress(report)
        self.active.pop(key)
        if self.fail:
            raise self.fail
        self.installed.add(key)
        return f"/m/{key}"

    def cancel(self, key):
        self.calls.append(("cancel", key))
        return True

    def delete(self, key):
        self.calls.append(("delete", key))
        self.installed.discard(key)
        return 1

    def load(self, key):
        self.calls.append(("load", key))
        self._loaded.append(key)

    def unload(self, key):
        self.calls.append(("unload", key))
        if key in self._loaded:
            self._loaded.remove(key)

    def loaded(self):
        return list(self._loaded)

    def disk_usage(self):
        return {"total_bytes": 123}

    def spec_for_inpaint_method(self, method):
        return next((s for s in self.specs_list if s.selector == method), None)

    def spec_for_detector_variant(self, name):
        return next((s for s in self.specs_list if s.selector == name), None)

    @staticmethod
    def format_size(n):
        return f"{n // (1024 * 1024)} MB"

    def missing_models(self, config):
        return [s for s in self.specs_list if s.key not in self.installed]


def test_model_manager_maps_the_registry_states(monkeypatch):
    fake = FakeModels()
    monkeypatch.setitem(sys.modules, "manga_models", fake)
    manager = svc.ModelManager()
    entries = {e.id: e for e in manager.entries()}
    assert entries["det"].status == "missing" and entries["det"].size_label == "11 MB"
    assert entries["det"].chip == "Not downloaded"
    fake.partial["det"] = 11 * 1024 * 1024 // 4
    assert "Paused at 25%" in manager.status("det").error
    seen: list = []
    entry = manager.download("det", seen.append)
    assert entry.status == "ready" and entry.chip == "Downloaded"
    assert seen and seen[0].status == "downloading" and seen[0].progress == 0.5
    assert manager.can_load("det") and not manager.can_load("aot_onnx")
    assert manager.load("det").status == "loaded"
    assert manager.unload("det").status == "ready"
    fake.fail = RuntimeError("HTTP 503")
    failed = manager.download("aot_onnx")
    assert failed.status == "error" and failed.error == "HTTP 503" and failed.chip == "Failed"
    fake.fail = type("DownloadCancelled", (Exception,), {})("stop")
    assert manager.download("aot_onnx").error.startswith("Cancelled")
    assert manager.cancel("aot_onnx") and manager.delete("det").status == "missing"
    assert manager.disk_usage() == 123 and manager.for_local_method("aot_onnx") == "aot_onnx"
    assert manager.required({}) == ["det", "aot_onnx"]
    assert manager.status("nope").status == "unavailable"
    monkeypatch.setitem(sys.modules, "manga_models", None)
    assert not svc.ModelManager().available and svc.ModelManager().entries() == []


@pytest.mark.skipif(not _has("manga_models"), reason="manga_models not importable")
def test_real_registry_statuses_stay_in_the_tmp_caches(iso):
    manager = svc.ModelManager()
    entries = manager.entries()
    assert {e.kind for e in entries} == {"detector", "inpaint"}
    assert all(e.status == "missing" for e in entries)  # nothing downloaded in the tmp caches
    assert all(str(iso["models"]) in e.path for e in entries)
    # the desktop dialog's three RT-DETR "ONNX Export" values
    assert {v[0] for v in manager.detector_variants()} == {"detector.onnx", "detector_int8.onnx",
                                                           "detector-v4-s_int8.onnx"}


# ==========================================================================
# Job adapters
# ==========================================================================


class FakeRunner:
    """``manga_runner.HeadlessMangaRunner`` contract."""

    instances: list = []
    summary: dict = {}
    hold: threading.Event = None

    def __init__(self, main_gui, **kwargs):
        self.main_gui = main_gui
        self.kwargs = kwargs
        self.stops: list = []
        FakeRunner.instances.append(self)

    def run(self, files=None, glossary_only=None, **_):
        progress = self.kwargs["progress"]
        progress(1, 2, label="Translating p1.png", failed=0)
        if FakeRunner.hold is not None:
            FakeRunner.hold.wait(5)
        return dict(FakeRunner.summary)

    def request_stop(self, graceful=None, force=False):
        self.stops.append((graceful, force))


class FakeRunError(RuntimeError):
    pass


def _runner_module(**summary):
    FakeRunner.instances = []
    FakeRunner.summary = summary
    FakeRunner.hold = None
    return _module("manga_runner", HeadlessMangaRunner=FakeRunner, MangaRunError=FakeRunError)


def test_batch_adapter_runs_the_headless_runner_with_the_selection(tmp_path, monkeypatch):
    pages = [write_png(tmp_path / "in" / f"{n}.png") for n in (1, 2)]
    out = tmp_path / "Output" / "1_translated" / "1.png"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"png")
    cbz = tmp_path / "in_translated.cbz"
    cbz.write_bytes(b"PK")
    monkeypatch.setitem(sys.modules, "manga_runner", _runner_module(
        ok=True, completed=1, failed=1, total=2, outputs=[str(out)], cbz_paths=[str(cbz)], stopped=False, error="",
        glossary_path="", glossary_only=False))
    session = types.SimpleNamespace(image_state_manager=object())
    token = svc.register_editor_session(session)
    owner = types.SimpleNamespace(config={})
    ctx = FakeCtx(owner, params={"files": pages, "run_files": pages, "image_range": "1-2", "skipped": ["k"],
                                 "folder_roots": [str(tmp_path / "in")], "split_first_level": True,
                                 "cbz_jobs": {}, "cbz_image_to_job": {}, "glossary_only": False,
                                 "output_root": str(tmp_path / "Output"), "editor_session": token},
                  inputs=pages)
    result = manga_kind.run_batch(ctx)
    runner = FakeRunner.instances[0]
    assert runner.main_gui is owner and runner.kwargs["host"] is ctx.host
    assert runner.kwargs["files"] == pages and runner.kwargs["image_range"] == "1-2"
    assert runner.kwargs["skipped"] == ["k"] and runner.kwargs["split_first_level"] is True
    assert runner.kwargs["image_state_manager"] is session.image_state_manager
    assert runner.kwargs["output_root"] == str(tmp_path / "Output")
    assert result == {"ok": True, "outputs": [str(out), str(cbz)]}
    assert ctx.results["manga_completed"] == 1 and ctx.results["manga_failed"] == 1
    assert ctx.results["manga_cbz"] == [str(cbz)] and ctx.outputs == [str(out), str(cbz)]
    progress = [data for kind, data in ctx.events if kind == "progress"]
    assert progress[0]["completed"] == 0 and progress[1]["label"] == "Translating p1.png"
    assert progress[-1]["completed"] == 1 and progress[-1]["failed"] == 1
    assert ctx.output_dir == str(out.parent)
    # an imported OCR JSON goes to the runner before Start (manga_env.import_ocr_session, desktop batch import)
    imports: list = []
    monkeypatch.setitem(sys.modules, "manga_env", _module(
        "manga_env", import_ocr_session=lambda state, path=None, *, document=None, files=None:
        imports.append((state, path, list(files))) or {p: {} for p in files[:1]}))
    ocr = tmp_path / "session.json"
    ocr.write_text("{}", encoding="utf-8")
    ctx = FakeCtx(owner, params={"files": pages, "run_files": pages[1:], "imported_ocr": str(ocr)}, inputs=pages)
    manga_kind.run_batch(ctx)
    assert imports == [(FakeRunner.instances[-1], str(ocr), pages[1:])] and not ctx.logs
    ctx = FakeCtx(owner, params={"files": pages, "imported_ocr": str(tmp_path / "gone.json")}, inputs=pages)
    assert manga_kind.run_batch(ctx)["ok"] is True and "Imported OCR not reused" in ctx.logs[0]
    # failures and a refused start
    monkeypatch.setitem(sys.modules, "manga_runner", _runner_module(ok=False, completed=0, failed=2, total=2,
                                                                    outputs=[], cbz_paths=[], error="", stopped=False))
    assert manga_kind.run_batch(FakeCtx(params={"files": pages}))["error"] == "2 page(s) failed"

    class Refusing(FakeRunner):
        def run(self, **_):
            raise FakeRunError("The image range does not include any loaded rows.")

    monkeypatch.setitem(sys.modules, "manga_runner", _module("manga_runner", HeadlessMangaRunner=Refusing,
                                                             MangaRunError=FakeRunError))
    with pytest.raises(JobError, match="image range"):
        manga_kind.run_batch(FakeCtx(params={"files": pages}))
    with pytest.raises(JobError, match="Missing image"):
        manga_kind.run_batch(FakeCtx(params={"files": [str(tmp_path / "gone.png")]}))
    monkeypatch.setitem(sys.modules, "manga_runner", None)
    with pytest.raises(JobError, match="manga_runner"):
        manga_kind.run_batch(FakeCtx(params={"files": pages}))


def test_batch_stop_bridge_forwards_graceful_immediate_and_force(tmp_path, monkeypatch):
    pages = [write_png(tmp_path / "in" / "1.png")]
    monkeypatch.setitem(sys.modules, "manga_runner", _runner_module(ok=True, completed=0, failed=0, total=1,
                                                                    outputs=[], cbz_paths=[], stopped=True))
    monkeypatch.setattr(manga_kind, "_STOP_POLL", 0.01)
    FakeRunner.hold = threading.Event()
    ctx = FakeCtx(params={"files": pages})
    result: dict = {}
    thread = threading.Thread(target=lambda: result.update(manga_kind.run_batch(ctx)))
    thread.start()
    _jobs_helpers().wait_for(lambda: FakeRunner.instances)
    runner = FakeRunner.instances[0]
    ctx.request("graceful")
    assert _jobs_helpers().wait_for(lambda: runner.stops == [(True, False)])
    ctx.request("force")
    assert _jobs_helpers().wait_for(lambda: runner.stops == [(True, False), (None, True)])
    FakeRunner.hold.set()
    thread.join(5)
    assert result["ok"] is None
    # an immediate first stop (graceful_stop off)
    FakeRunner.instances = []
    FakeRunner.hold = threading.Event()
    ctx = FakeCtx(params={"files": pages})
    thread = threading.Thread(target=lambda: manga_kind.run_batch(ctx))
    thread.start()
    _jobs_helpers().wait_for(lambda: FakeRunner.instances)
    ctx.request("immediate")
    assert _jobs_helpers().wait_for(lambda: FakeRunner.instances[0].stops == [(False, False)])
    FakeRunner.hold.set()
    thread.join(5)


class FakeEditorSession:
    """``manga_editor_core.MangaEditorSession`` contract (box model + workflow methods)."""

    def __init__(self, image_paths=()):
        self.calls: list = []
        self.pages = list(image_paths)
        self.current_page = None
        self.boxes: dict = {}
        self.output_revision = 0
        self._log_callback = None
        self.rendered: dict = {}
        self.image_state_manager = types.SimpleNamespace(name="state")
        self.hold = None

    def set_pages(self, pages):
        self.pages = list(pages)

    def open_page(self, path):
        self.calls.append(("open_page", path))
        self.current_page = path
        return self.page_snapshot(path)

    def page_snapshot(self, path=None):
        path = path or self.current_page
        boxes = [dict(b, index=i) for i, b in enumerate(self.boxes.get(path, []))]
        return {"image_path": path, "boxes": boxes, "cleaned_path": None,
                "rendered_path": self.rendered.get(path), "translated_path": None, "revision": self.output_revision}

    def add_box(self, x, y, w, h, *, shape="rect", polygon=None):
        self.calls.append(("add_box", round(x), round(y), round(w), round(h), shape, polygon is not None))
        self.boxes.setdefault(self.current_page, []).append(
            {"x": x, "y": y, "width": w, "height": h, "shape": shape, "polygon": polygon, "ocr_text": "",
             "translation": "", "exclude_from_clean": False, "free_text": False, "inpaint_iterations": None})

    def update_box(self, index, x, y, w, h, *, owner=None, rerender=True):
        self.calls.append(("update_box", index, round(x), round(y), round(w), round(h), rerender))
        self.boxes[self.current_page][index].update(x=x, y=y, width=w, height=h)

    def delete_box(self, index):
        self.calls.append(("delete_box", index))
        del self.boxes[self.current_page][index]

    def set_box_excluded(self, index, value):
        self.calls.append(("set_box_excluded", index, value))
        self.boxes[self.current_page][index]["exclude_from_clean"] = value

    def set_box_free_text(self, index, value):
        self.calls.append(("set_box_free_text", index, value))
        self.boxes[self.current_page][index]["free_text"] = value

    def set_box_iterations(self, index, value):
        self.calls.append(("set_box_iterations", index, value))

    def edit_box_text(self, index, *, ocr_text=None, translation=None):
        self.calls.append(("edit_box_text", index, ocr_text, translation))
        box = self.boxes[self.current_page][index]
        if ocr_text is not None:
            box["ocr_text"] = ocr_text
        if translation is not None:
            box["translation"] = translation
        return True

    def _step(self, name, image=None, owner=None):
        self.calls.append((name, image, owner))
        if self.hold is not None:
            self.hold.wait(5)

    def detect(self, image=None, *, owner=None):
        self._step("detect", image, owner)
        self.boxes[image] = [{"x": 1, "y": 2, "width": 10, "height": 12, "shape": "rect", "ocr_text": "",
                              "translation": "", "exclude_from_clean": False, "free_text": False}]
        return [{"bbox": [1, 2, 10, 12]}]

    def clean(self, image=None, *, owner=None):
        self._step("clean", image, owner)

    def recognize(self, image=None, *, owner=None):
        self._step("recognize", image, owner)

    def translate(self, image=None, *, owner=None):
        self._step("translate", image, owner)
        self.rendered[image] = image
        self.output_revision += 1
        return {"rendered_path": image}

    def translate_all(self, images=None, *, owner=None):
        self._step("translate_all", tuple(images or ()), owner)
        return {path: path for path in images or ()}

    def save_and_update_overlay(self, image=None, *, owner=None):
        self._step("render", image, owner)

    def import_ocr(self, path, images=None, *, owner=None, render=True):
        self._step("import_ocr", path, owner)
        return {"matched": 1, "files": len(images or ()), "translated_regions": 2, "rendered": {}}

    def ocr_box(self, index, *, owner=None):
        self._step("ocr_box", index, owner)
        return "テキスト"

    def translate_box(self, index, *, owner=None):
        self._step("translate_box", index, owner)
        return "Text"

    def clean_box(self, index, *, owner=None):
        self._step("clean_box", index, owner)

    def stop(self, force=False):
        self.calls.append(("stop", force))
        if self.hold is not None:
            self.hold.set()

    def export_ocr(self, destination, image_paths=None, *, source_root=None):
        self.calls.append(("export_ocr", destination, tuple(image_paths or ())))
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_text("{}", encoding="utf-8")
        return {"path": destination, "pages": 1, "translated_regions": 0}


def test_step_adapter_calls_the_session_with_the_job_owner(tmp_path, monkeypatch):
    page = write_png(tmp_path / "p" / "page.png")
    other = write_png(tmp_path / "p" / "other.png")
    session = FakeEditorSession([page, other])
    token = svc.register_editor_session(session)
    owner = types.SimpleNamespace(config={})
    for step in ("detect", "clean", "recognize", "render"):
        ctx = FakeCtx(owner, params=svc.step_spec(step, token, page).params)
        assert manga_kind.run_step(ctx)["ok"] is True
        assert session.calls[-1] == (step, page, owner)
        assert ctx.results["manga_step"] == step and ("manga_step", {"step": step, "image": page}) in ctx.events
        assert ctx.outputs == []  # nothing rendered yet in the fake
    ctx = FakeCtx(owner, params=svc.step_spec("translate", token, page).params)
    manga_kind.run_step(ctx)
    assert ctx.outputs == [page] and ctx.results["manga_revision"] == session.output_revision
    ctx = FakeCtx(owner, params=svc.step_spec("translate_all", token, page, images=[page, other]).params)
    manga_kind.run_step(ctx)
    assert session.calls[-1] == ("translate_all", (page, other), owner) and ctx.outputs == [page, other]
    session.current_page = other
    ctx = FakeCtx(owner, params=svc.step_spec("ocr_box", token, page, index=0).params)
    manga_kind.run_step(ctx)
    assert session.calls[-2:] == [("open_page", page), ("ocr_box", 0, owner)]
    assert ctx.results["manga_box_text"] == "テキスト"
    ocr = tmp_path / "ocr.json"
    ocr.write_text("{}", encoding="utf-8")
    ctx = FakeCtx(owner, params=svc.step_spec("import_ocr", token, page, images=[page], extra={"path": str(ocr)}).params)
    manga_kind.run_step(ctx)
    assert ctx.results["manga_import"] == {"matched": 1, "files": 1, "translated_regions": 2}
    # the session's log lines go to the job log while the step runs, then back
    session._log_callback = None

    def logging_detect(image=None, *, owner=None):
        session._log_callback("🔍 detecting", "info")
        return []

    monkeypatch.setattr(session, "detect", logging_detect)
    ctx = FakeCtx(owner, params=svc.step_spec("detect", token, page).params)
    manga_kind.run_step(ctx)
    assert "🔍 detecting" in ctx.logs and session._log_callback is None
    with pytest.raises(JobError, match="editor session is gone"):
        manga_kind.run_step(FakeCtx(params=svc.step_spec("detect", "es-missing", page).params))
    with pytest.raises(JobError, match="OCR file is missing"):
        manga_kind.run_step(FakeCtx(params=svc.step_spec("import_ocr", token, page, extra={"path": "x"}).params))
    with pytest.raises(JobError, match="Unknown manga editor step"):
        manga_kind.run_step(FakeCtx(params={"step": "nope", "session": token, "image": page}))


def test_step_stop_reaches_the_session(tmp_path, monkeypatch):
    page = write_png(tmp_path / "page.png")
    session = FakeEditorSession([page])
    session.hold = threading.Event()
    token = svc.register_editor_session(session)
    monkeypatch.setattr(manga_kind, "_STOP_POLL", 0.01)
    ctx = FakeCtx(params=svc.step_spec("translate", token, page).params)
    result: dict = {}
    thread = threading.Thread(target=lambda: result.update(manga_kind.run_step(ctx)))
    thread.start()
    _jobs_helpers().wait_for(lambda: any(c[0] == "translate" for c in session.calls))
    ctx.request("graceful")
    thread.join(5)
    assert ("stop", False) in session.calls and result["ok"] is None


def test_manga_kinds_run_through_the_job_service(tmp_path, monkeypatch, manga_kinds):
    tj = _jobs_helpers()
    page = write_png(tmp_path / "in" / "1.png")
    monkeypatch.setitem(sys.modules, "manga_runner", _runner_module(
        ok=True, completed=1, failed=0, total=1, outputs=[page], cbz_paths=[], stopped=False, error=""))
    service, backend = tj.make_service(tmp_path)
    job_id = service.submit(JobSpec("manga", "Series · 1 image", inputs=(page,),
                                    params={"files": [page], "run_files": [page]}, resumable=False))
    assert service.wait_idle(tj.TIMEOUT)
    snap = service.snapshot(job_id)
    assert snap.state is JobState.DONE, snap.error
    assert snap.result["manga_outputs"] == [page] and snap.progress.completed == 1
    assert ("reset", "translation") in backend.events
    assert FakeRunner.instances[0].main_gui is backend.owners[-1]
    session = FakeEditorSession([page])
    token = svc.register_editor_session(session)
    job_id = service.submit(svc.step_spec("detect", token, page))
    assert service.wait_idle(tj.TIMEOUT)
    snap = service.snapshot(job_id)
    assert snap.state is JobState.DONE, snap.error
    assert session.calls[-1][0] == "detect" and session.calls[-1][2] is backend.owners[-1]
    assert job_kinds.get_kind("manga").verb == "Translating manga" and not job_kinds.get_kind("manga").resumable
    service.close()


# ==========================================================================
# Intents, quick chips, the feature
# ==========================================================================


def test_intent_router_offers_the_manga_translator_for_images_and_archives(tmp_path):
    from glossarion_mobile.services.files import ImportedFile
    from glossarion_mobile.services.intents import ACTION_MANGA, MANGA_REASON, IntentImport, IntentRouter

    def imp(name):
        return IntentImport(item={"path": name}, imported=ImportedFile(path=str(tmp_path / name), name=name, size=1,
                                                                        source=name, target="inbox"))

    bare = IntentRouter(files=types.SimpleNamespace())
    assert {a.id: a for a in bare.actions_for(imp("p.png"))}[ACTION_MANGA].disabled_reason == MANGA_REASON
    assert ACTION_MANGA not in {a.id for a in bare.actions_for(imp("book.epub"))}
    got = []
    router = IntentRouter(files=types.SimpleNamespace(), handlers={ACTION_MANGA: got.append})
    for name in ("p.png", "vol.cbz", "pages.zip", "x.webp"):
        assert {a.id: a for a in router.actions_for(imp(name))}[ACTION_MANGA].disabled_reason is None
    asyncio.run(router.perform(ACTION_MANGA, imp("p.png")))
    assert got and got[0].imported.name == "p.png"


def test_quick_chips_for_attachments():
    from glossarion_mobile.ui.chat.quick_chips import chips_for_attachment

    assert chips_for_attachment(None) == []
    assert chips_for_attachment("a/page.png") == ["translate", "manga"]
    assert chips_for_attachment("a/vol.CBZ") == ["translate", "manga"]
    assert chips_for_attachment("a/book.epub") == ["translate", "extract_glossary", "open_reader"]
    assert chips_for_attachment("a/notes.txt") == ["translate", "extract_glossary", "open_reader"]


@needs_flet
def test_quick_chip_row_rebuilds_only_on_change_and_dismisses():
    from glossarion_mobile.ui.chat.quick_chips import QuickChips

    picked = []
    chips = QuickChips(on_select=picked.append)
    assert chips.set_attachment("p.png") and chips.visible_ids == ("translate", "manga")
    assert not chips.set_attachment("p.png")
    chips.row.controls[1].on_click(None)
    assert picked == ["manga"]
    chips.dismiss("p.png")
    assert chips.visible_ids == () and not chips.control.visible
    assert chips.set_attachment("q.png") and chips.visible_ids == ("translate", "manga")
    assert chips.set_attachment("q.png", hidden=True) and chips.visible_ids == ()


@needs_flet
def test_chat_quick_chips_follow_the_attachment_and_route_to_the_tools():
    from glossarion_mobile.ui.chat.chat_view import ChatView
    from glossarion_mobile.ui.chat.quick_chips import QuickChips
    from glossarion_mobile.ui.chat.send_state import SendAction, SendState

    calls: list = []
    view = types.SimpleNamespace(
        composer=types.SimpleNamespace(attachment={"path": "/x/book.epub"}),
        _on_tool=lambda tool: calls.append(("tool", tool)),
        on_send_action=lambda action: calls.append(("send", action)),
        env=types.SimpleNamespace(open_reader=lambda folder, path: calls.append(("reader", folder, path))),
        notify=lambda message: calls.append(("notify", message)),
        _spawn=lambda coro: calls.append(("spawn", coro)))
    view.quick_chips = QuickChips(on_select=lambda cid: ChatView._on_quick_chip(view, cid))
    assert ChatView._refresh_quick_chips(view, SendState.IDLE_READY)
    assert view.quick_chips.visible_ids == ("translate", "extract_glossary", "open_reader")
    assert not ChatView._refresh_quick_chips(view, SendState.IDLE_READY)  # unchanged: nothing to push
    assert ChatView._refresh_quick_chips(view, SendState.RUNNING) and view.quick_chips.visible_ids == ()
    view.composer.attachment = {"path": "/x/vol1.cbz"}
    ChatView._refresh_quick_chips(view, SendState.IDLE_READY)
    assert view.quick_chips.visible_ids == ("translate", "manga")
    view.composer.attachment = {"path": "/x/book.epub"}
    for chip in ("translate", "extract_glossary", "manga", "open_reader", "nope"):
        ChatView._on_quick_chip(view, chip)
    assert calls == [("send", SendAction.SEND), ("tool", "extract_glossary"), ("tool", "manga"),
                     ("reader", "", "/x/book.epub"), ("notify", "This action is not available here")]
    view.env = None
    ChatView._on_quick_chip(view, "open_reader")
    assert calls[-1] == ("notify", "The Reader is not available in this session")


class FakeShell:
    def __init__(self):
        self.screen_factory = lambda match: ("fallback", match.name)
        self.top_screen = None
        self.tablet = False


class FakeChatView:
    def __init__(self, attachment=None):
        self.tools: list = []
        self.composer = types.SimpleNamespace(attachment=attachment, set_plus_open=lambda value: None)

    def _on_tool(self, tool_id):
        self.tools.append(tool_id)


def _fake_app(tmp_path):
    from glossarion_mobile.services.intents import IntentRouter

    navigated = []
    app = types.SimpleNamespace(
        page=None, dispatcher=None, shell=FakeShell(), intents=IntentRouter(files=types.SimpleNamespace()),
        chat_view=FakeChatView({"path": str(tmp_path / "page.png")}), config_store=None,
        paths=types.SimpleNamespace(data=str(tmp_path / "data"), output=str(tmp_path / "Output")),
        navigate_to=lambda name, params=None, query=None: navigated.append((name, params, query)))
    app.navigated = navigated
    return app


def test_feature_wires_the_screen_intent_and_chat_tool(tmp_path):
    from glossarion_mobile.services.intents import ACTION_MANGA
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.manga.feature import MangaFeature, accepts_path

    app = _fake_app(tmp_path)
    feature = asyncio.run(MangaFeature.install(app))
    assert app.manga is feature and ACTION_MANGA in app.intents.handlers
    assert app.shell.screen_factory(parse_route("/tools/qa")) == ("fallback", "tools.qa")
    # ＋ › Manga translator / "Translate as manga": the composer attachment goes to the Files tab
    app.chat_view._on_tool("manga")
    assert feature.session.pending == [str(tmp_path / "page.png")]
    assert app.navigated[-1] == ("tools.manga", None, {"tab": "files"})
    app.chat_view._on_tool("qa")
    assert app.chat_view.tools == ["qa"]  # other tools reach the chat's own handler
    app.chat_view.composer.attachment = {"path": str(tmp_path / "book.epub")}
    app.chat_view._on_tool("manga")
    assert feature.session.pending == [str(tmp_path / "page.png")]  # an EPUB is not handed over
    imp = types.SimpleNamespace(imported=types.SimpleNamespace(path=str(tmp_path / "vol.cbz")))
    assert feature._from_intent(imp) == 1 and feature.session.pending[-1].endswith("vol.cbz")
    assert accepts_path("x.JPG") and not accepts_path("x.epub") and not accepts_path(None)
    assert feature.session.root == os.path.join(str(tmp_path / "data"), "manga")
    asyncio.run(MangaFeature.install(app))  # a second install does not wrap the chat twice
    app.chat_view._on_tool("qa")
    assert app.chat_view.tools == ["qa", "qa"]


# ==========================================================================
# Screens (in-memory Flet session)
# ==========================================================================


class FakeJobs:
    """The JobsFeature surface the manga screens use."""

    def __init__(self, kinds=("manga", "manga_step")) -> None:
        self.kinds = set(kinds)
        self.specs: list = []
        self.listeners: list = []
        self.view_listeners: list = []
        self.stops: list = []
        self.snaps: dict = {}

    def has_kind(self, kind):
        return kind in self.kinds

    async def submit(self, spec):
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

    def subscribe(self, callback, immediate=False):
        self.view_listeners.append(callback)
        return lambda: self.view_listeners.remove(callback) if callback in self.view_listeners else None

    def log_buffer(self, job_id):
        from glossarion_mobile.services.dispatcher import LogBuffer

        return LogBuffer(100, name=f"job:{job_id}")

    def request_stop(self, job_id=None, **kwargs):
        self.stops.append(job_id)
        return "graceful"

    def finish(self, job_id, *, state=JobState.DONE, outputs=(), result=None, error=None):
        import dataclasses

        snap = dataclasses.replace(self.snaps[job_id], state=state, started=2.0, finished=3.0, outputs=tuple(outputs),
                                   result=dict(result or {}), error=error)
        self.snaps[job_id] = snap
        for callback in list(self.listeners):
            callback(snap, JobState.RUNNING)
        return snap


class FakeFiles:
    def __init__(self, picks=None, folder=None):
        self.picks = list(picks or [])
        self.folder = folder
        self.exports: list = []

    async def pick_files(self, **kwargs):
        self.last_kwargs = kwargs
        paths = self.picks.pop(0) if self.picks else []
        return [types.SimpleNamespace(path=p, name=os.path.basename(p)) for p in paths]

    async def pick_folder(self, **kwargs):
        if isinstance(self.folder, Exception):
            raise self.folder
        return types.SimpleNamespace(path=self.folder)

    def export_options(self, path):
        from glossarion_mobile.services.files import ExportOption

        return [ExportOption("share", "Share…", "IOS_SHARE"), ExportOption("save", "Save to…", "SAVE_ALT")]

    async def export(self, option_id, path, confirmed=False):
        self.exports.append((option_id, path))
        return True

    async def share(self, paths, **kwargs):
        self.exports.append(("share", list(paths)))
        return True


class FakePrefs:
    def __init__(self):
        self.data: dict = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


def _ctx(page, store, *, jobs=None, files=None, settings=None, output_root="", data_dir="", tablet=False):
    from glossarion_mobile.ui.tools.common import ToolsContext

    notes: list = []
    navigated: list = []
    ctx = ToolsContext(service=None, page=page, navigate=lambda n, p=None, q=None: navigated.append((n, p, q)),
                       notify=lambda message, action=None, on_action=None: notes.append((message, action, on_action)),
                       jobs=jobs if jobs is not None else FakeJobs(), files=files, prefs=FakePrefs(),
                       platform="android", store=store, settings=settings, output_root=output_root,
                       data_dir=data_dir, tablet=tablet)
    ctx.notes, ctx.navigated = notes, navigated
    ctx.extras["answers"] = []
    return ctx


def _mount(page, body):
    page.views[0].controls.append(body)
    page.update()


async def _settle(times=10):
    for _ in range(times):
        await asyncio.sleep(0.01)


def _session(tmp_path, store):
    from glossarion_mobile.ui.tools.manga.feature import MangaSession

    def save(updates):
        for key, value in updates.items():
            if isinstance(store, dict):
                store[key] = value

    return MangaSession(data_dir=str(tmp_path / "data"), config_source=lambda: dict(store), save=save)


@needs_flet
@needs_cores
def test_files_tab_adds_sorts_skips_and_starts_a_batch(iso, tmp_path):
    from glossarion_mobile.services.files import FolderPickUnavailable
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    fixture = make_series(tmp_path)
    store: dict = {}

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        jobs = FakeJobs()
        files = FakeFiles(picks=[[fixture["cbz"]]], folder=FolderPickUnavailable("Android does not let apps read a "
                                                                                 "picked folder directly"))
        ctx = _ctx(page, store, jobs=jobs, files=files, output_root=str(iso["Output"]))
        manga = _session(tmp_path, store)
        screen = MangaScreen(parse_route("/tools/manga?tab=files"), ctx, session=manga)
        _mount(page, screen.get_body())
        screen.did_show()
        await _settle()
        tab = screen.files_tab
        assert manga.loaded and tab.start_button.disabled and tab.empty.visible
        # the desktop checkboxes' defaults (manga_settings_defaults), not "off"
        assert tab.create_cbz_switch.value is True and tab.consolidate_switch.value is True
        # Add folder -> SAF refusal -> "Pick a .zip instead"
        assert await tab.pick_folder() == 0
        message, action, on_action = ctx.notes[-1]
        assert "Pick a .zip instead" in message and action == "Pick a .zip instead" and callable(on_action)
        assert await tab.pick_archive() == 3
        assert files.last_kwargs["allowed_extensions"] == ["cbz", "zip"]
        assert await tab.add_paths([fixture["series"]]) == 6
        assert len(tab.list_view.controls) == 9 and not tab.start_button.disabled
        assert store["manga_selected_files"] == manga.files.files
        first = manga.files.files[0]
        await tab._mutate(manga.files.toggle_skip, first)
        assert manga.files.is_skipped(first) and tab.summary.value == "9 images · 8 will run"
        tab.range_field.value = "2-3"
        tab._on_range()
        assert tab.range_status.value == "2 in range · 2 of 9 will run"
        tab.range_field.value = ""
        tab._on_range()
        tab.sort_buttons.selected = ["name"]
        tab._on_sort()
        await _settle()
        assert os.path.basename(manga.files.files[0]) == "1.png" and tab.sort_buttons.selected == ["name"]
        tab.split_switch.value = True
        tab._on_split()
        assert tab.group_picker.visible and len(tab.group_picker.options) == 4  # All + vol1 + ch1 + ch2
        tab.group_picker.value = "1"
        tab._on_group()
        assert len(tab.list_view.controls) == 3 and tab.list_view.show_default_drag_handles is False
        ctx.extras["answers"] = ["run"]  # the models are not downloaded in the tmp caches: Start anyway
        job_id = await tab.start()
        if manga.models.available:
            assert ctx.extras["asked"][-1][0] == "Download models"
        spec = jobs.specs[-1]
        assert spec.kind == "manga" and spec.params["files"] == manga.files.files  # all groups run
        assert len(spec.inputs) == 8 and manga.batch_job_id == job_id
        from glossarion_mobile.ui.components.log_console import LogConsole

        assert isinstance(tab.console_holder.content, LogConsole) and tab.console_job == job_id
        assert tab.start_button.disabled and tab.stop_button.visible
        tab._on_stop()
        assert jobs.stops == [job_id]  # Stop reaches the JobService (a queued job is cancelled there)
        out = tmp_path / "Output" / "x_translated" / "x.png"
        out.parent.mkdir(parents=True)
        out.write_bytes(b"png")
        jobs.finish(job_id, result={"manga_outputs": [str(out)], "manga_completed": 7, "manga_failed": 1,
                                    "manga_cbz": [str(tmp_path / "vol1_translated.cbz")]})
        await _settle()
        assert tab.run_status.value.startswith("Done · 7 translated, 1 failed · CBZ: vol1_translated.cbz")
        assert manga.last_outputs == [str(out)] and not tab.start_button.disabled
        path = await tab.download_images()
        assert path == str(out) and ctx.extras["manga_last_sheet"].title == "x.png"
        # Create CBZ: the desktop button's packing of the run's <page>_translated folders
        write_png(iso["Output"] / "1_translated" / "1.png")
        cbz = await tab.create_cbz()
        assert cbz == str(iso["Output"] / "Output_translated.cbz")
        with zipfile.ZipFile(cbz) as archive:
            assert archive.namelist() == ["1.png"]
        assert ctx.extras["manga_last_sheet"].title == "Output_translated.cbz"
        # selection mode: long-press, remove selected
        tab.group_picker.value = ""
        tab._on_group()
        target = manga.files.files[-1]
        tab._on_row_long_press(target)
        assert tab.selection_mode and tab.selection_bar.visible
        assert await tab.remove_selected() == 1 and target not in manga.files.files
        ctx.extras["answers"] = ["yes"]
        assert await tab.clear_all() and manga.files.files == [] and store["manga_selected_files"] == []
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
@needs_cores
def test_files_tab_glossary_only_and_missing_kind(iso, tmp_path):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    fixture = make_series(tmp_path)

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        store: dict = {}
        jobs = FakeJobs(kinds=())
        ctx = _ctx(page, store, jobs=jobs, output_root=str(iso["Output"]))
        manga = _session(tmp_path, store)
        screen = MangaScreen(parse_route("/tools/manga"), ctx, session=manga)
        _mount(page, screen.get_body())
        screen.did_show()
        await _settle()
        await screen.files_tab.add_paths([fixture["series"]])
        assert screen.files_tab.start_button.disabled
        assert screen.files_tab.start_button.tooltip == "Manga jobs are not available in this build"
        assert await screen.files_tab.start() is None and jobs.specs == []
        jobs.kinds = {"manga"}
        if manga.models.available:
            ctx.extras["answers"] = [None]  # the detector download question, dismissed: no job
            assert await screen.files_tab.start(glossary_only=True) is None and jobs.specs == []
        ctx.extras["answers"] = ["run"]
        await screen.files_tab.start(glossary_only=True)
        assert jobs.specs[-1].params["glossary_only"] is True
        # handed-over files land in the list on the next show
        manga.pending.append(fixture["cbz"])
        assert await screen.take_pending() == 3 and manga.pending == []
        screen.dispose()

    asyncio.run(scenario())


def _settings_store(tmp_path):
    from glossarion_mobile.state.config_store import MobileConfigStore
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    schema = SchemaAccess()
    config_path = tmp_path / "config.json"
    store = MobileConfigStore(config_path, debounce=10, defaults=schema.effective_default,
                              reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8"))
                              if Path(p).exists() else {},
                              writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk),
                                                                                        encoding="utf-8"))
    store.load()
    return store, schema


def _chips_under_disabled(root, seen: list | None = None) -> list:
    """ReasonChips with a disabled ancestor (content / controls / title / subtitle / leading / trailing / tabs);
    ``seen`` collects every chip reached."""
    import flet as ft

    from glossarion_mobile.ui.components.reason_chip import ReasonChip

    found: list = []

    def walk(control, disabled):
        if isinstance(control, ReasonChip):
            if seen is not None:
                seen.append(control.reason)
            if disabled:
                found.append(control.reason)
        disabled = disabled or bool(getattr(control, "disabled", False))
        for name in ("content", "controls", "title", "subtitle", "leading", "trailing", "tabs"):
            child = getattr(control, name, None)
            for kid in (child if isinstance(child, (list, tuple)) else [child]):
                if isinstance(kid, ft.Control):
                    walk(kid, disabled)

    walk(root, False)
    return found


@needs_flet
@pytest.mark.skipif(not _has("settings_schema"), reason="settings_schema not importable")
def test_settings_tab_providers_inpainting_rendering_and_schema_groups(iso, tmp_path, monkeypatch):
    from glossarion_mobile.ui.components.reason_chip import ReasonChip
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.settings.context import SettingsContext
    from glossarion_mobile.ui.tools.manga import settings as ms
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    monkeypatch.setitem(sys.modules, "manga_models", FakeModels())
    store, schema = _settings_store(tmp_path)

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        settings = SettingsContext(page=page, store=store, schema=schema)
        ctx = _ctx(page, store, settings=settings, output_root=str(iso["Output"]))
        manga = _session(tmp_path, {})
        screen = MangaScreen(parse_route("/tools/manga?tab=settings"), ctx, session=manga)
        _mount(page, screen.get_body())
        assert screen.current_tab == "settings"
        tab = screen.settings_tab
        tab.did_show()
        tiles = tab.provider_tiles
        assert list(tiles) == [v for v, _l in svc.OCR_PROVIDERS]
        # U9: the row stays enabled with no tap action, so its ReasonChip opens the reason (UI_SPEC §5.2)
        assert not tiles["manga-ocr"].disabled and tiles["manga-ocr"].on_click is None
        assert isinstance(tiles["manga-ocr"].trailing, ReasonChip)
        assert not tiles["custom-api"].disabled
        assert not tab.select_provider("easyocr") and store.get("manga_ocr_provider") is None
        assert tab.select_provider("azure") and store.get("manga_ocr_provider") == "azure"
        keys = [str(getattr(c, "key", "") or "").rsplit("-g", 1)[0] for c in tab.ocr_card.content.controls]
        assert "ms-azure-key" in keys and "ms-azure-endpoint" in keys
        tab.select_provider("custom-api")
        disable = next(c for c in tab.ocr_card.content.controls
                       if str(getattr(c, "key", "") or "").startswith("ms-ocr-no-thinking-g"))
        disable.value = False
        disable.on_change(types.SimpleNamespace(control=disable))
        assert store.get(("manga_settings", "ocr", "manga_ocr_disable_thinking")) is False
        # detection: the RT-DETR ONNX download row of the phone default
        row = tab.model_rows["detector"]
        assert row.model_id == "det" or row.model_id  # FakeModels' detector
        # inpainting
        assert not tab.select_inpaint("hybrid")
        assert tab.select_inpaint("local", "aot_onnx")
        assert store.get("manga_local_inpaint_model") == "aot_onnx"
        assert store.get(("manga_settings", "inpainting", "local_method")) == "aot_onnx"
        assert "inpaint" in tab.model_rows
        # U9 (UI_SPEC §5.2): no ReasonChip of the tab sits under a disabled control (Flet would disable it)
        seen: list = []
        assert _chips_under_disabled(screen.get_body(), seen) == [] and seen
        assert not tab.select_inpaint("local", "lama")  # torch JIT
        assert tab.select_inpaint("local", "custom-image-edit")
        tab.set_image_edit_endpoint(" https://img.example/v1 ")
        assert store.get("custom_image_edit_endpoint") == "https://img.example/v1"
        assert store.get("manga_custom-image-edit_model_path") == "https://img.example/v1"
        assert store.get("use_custom_image_edit_endpoint") is True
        assert tab.select_inpaint("skip") and store.get("manga_skip_inpainting") is True
        # rendering: preview follows the settings
        tab._set_and_preview("manga_text_color", [0, 0, 0])
        style = ms.preview_style(store.snapshot())
        assert style["color"] == "#000000" and style["text"] == ms.SAMPLE_TEXT.upper()
        tab._set_hex("manga_shadow_color", "#00ff00")
        assert store.get("manga_shadow_color") == [0, 255, 0]
        if svc.presets_available():
            # shared presets (manga_settings_defaults.font_preset_updates): measured once on the io pool
            # under JOB_LOCK (scratch manga tabs write os.environ), then cached for the session
            svc._PRESET_CACHE.clear()
            assert not tab.apply_preset("small")  # not measured yet: the button goes through the io pool
            from job_runner import JOB_LOCK

            with JOB_LOCK:  # a job owns the process state: the preset waits for it
                assert not await tab.apply_preset_async("small")
            assert ctx.notes[-1][0] == "Presets can be applied once the running job has finished"
            assert await tab.apply_preset_async("small")
            assert store.get(("manga_settings", "font_sizing", "algorithm")) == "conservative"
            assert store.get("manga_strict_text_wrapping") is True and store.get("manga_max_font_size") == 48
            assert svc.cached_font_preset_updates("small") is not None
            assert tab.apply_preset("small")  # cached: applied on the UI loop
        else:
            assert not await tab.apply_preset_async("small")
        # the shared editors (prompt / secret / service-account file)
        prompt = tab.edit_prompt("manga_ocr_prompt", "OCR prompt", "ocr")
        if _has("manga_env"):
            import manga_env

            assert prompt.default == manga_env.default_manga_ocr_prompt() and prompt.field.value == prompt.default
        prompt.field.value = "Read the bubble text."
        assert prompt.save() and store.get("manga_ocr_prompt") == "Read the bubble text."
        secret = tab.edit_secret("Azure Computer Vision key", "azure_vision_key")
        secret.field.value = "  azkey  "
        assert secret.save() and store.get("azure_vision_key") == "azkey"
        bad = tmp_path / "bad.json"
        bad.write_text("{}", encoding="utf-8")
        path_editor = tab.edit_google_credentials()
        path_editor.field.value = str(bad)
        assert not path_editor.save() and path_editor.error_text.value == "Invalid credentials JSON"
        good = tmp_path / "sa.json"
        good.write_text(json.dumps({"type": "service_account", "project_id": "p", "private_key": "k",
                                    "client_email": "e@x"}), encoding="utf-8")
        path_editor.field.value = str(good)
        assert path_editor.save()
        kept = store.get("google_vision_credentials")
        assert kept == os.path.join(manga.root, "credentials", "sa.json") and os.path.isfile(kept)
        image_edit = tab.edit_image_edit_prompts()
        image_edit.field.value = ""
        assert image_edit.save()
        if image_edit.default:
            assert store.get("custom_image_edit_system_prompt") == image_edit.default
        # the schema groups (built lazily)
        group_ids = list(tab.group_tiles)
        assert group_ids[0] == "ocr" and "advanced" in group_ids and "detection" in group_ids
        tile, keys = tab.group_tiles["advanced"]
        assert tile.controls == []
        tab._expand_group("advanced", keys)
        assert tile.controls and "advanced" in tab.group_built
        assert not set(keys) & ms.CURATED_KEYS
        screen.dispose()

    asyncio.run(scenario())


@pytest.mark.skipif(not _has("settings_schema"), reason="settings_schema not importable")
def test_schema_keys_group_by_the_desktop_tabs():
    import settings_schema
    from glossarion_mobile.ui.tools.manga.settings import group_manga_keys

    keys = settings_schema.section("manga.settings").keys
    groups = {gid: set(group_keys) for gid, _title, group_keys in group_manga_keys(keys)}
    assert sum(len(v) for v in groups.values()) == len(keys)
    assert "manga_settings.ocr.detector_type" in groups["detection"]
    assert "manga_settings.ocr.ocr_batch_size" in groups["ocr"]
    assert "manga_settings.advanced.hd_strategy" in groups["advanced"]
    assert "manga_settings.cloud_inpaint_prompt" in groups["inpainting"]
    assert "manga_shadow_blur" in groups["rendering"]
    assert all(k.startswith(("rapidocr_", "qwen2vl_", "manga_settings.ocr.", "manga_ocr"))
               for k in groups["ocr"])


@needs_flet
def test_model_rows_download_load_delete(tmp_path, monkeypatch):
    from glossarion_mobile.ui.tools.manga.models import ModelDownloadRow, ModelManagerSheet

    fake = FakeModels()
    monkeypatch.setitem(sys.modules, "manga_models", fake)

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        ctx = _ctx(page, {})
        manager = svc.ModelManager()
        row = ModelDownloadRow(ctx, manager, "det", key_prefix="t")
        assert row.download_button.visible and not row.delete_button.visible
        entry = await row.download()
        assert entry.status == "ready" and row.delete_button.visible and row.load_button.visible
        assert ctx.notes[-1][0] == "Downloaded Detector"
        assert (await row.toggle_load()).status == "loaded" and row.load_button.icon is not None
        assert (await row.toggle_load()).status == "ready"
        ctx.extras["answers"] = ["yes"]
        assert (await row.delete()).status == "missing"
        inpaint = ModelDownloadRow(ctx, manager, "aot_onnx", key_prefix="u")
        fake.installed.add("aot_onnx")
        inpaint.update()
        assert not inpaint.load_button.visible  # inpainters load when a run starts
        sheet = ModelManagerSheet(ctx, manager)
        assert set(sheet.rows) == {"det", "aot_onnx"} and sheet.usage.value == "On this device: 0 MB"

    asyncio.run(scenario())


@needs_flet
def test_runs_ask_to_download_missing_models_first(tmp_path, monkeypatch):
    from glossarion_mobile.ui.tools.manga import editor as me
    from glossarion_mobile.ui.tools.manga.models import ensure_models

    fake = FakeModels()
    monkeypatch.setitem(sys.modules, "manga_models", fake)

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        ctx = _ctx(session.page, {})
        manager = svc.ModelManager()
        ctx.extras["answers"] = [None]  # dismissed: nothing runs
        assert await ensure_models(ctx, manager, {}) is False and fake.calls == []
        title, body = ctx.extras["asked"][-1]
        assert title == "Download models" and body.startswith("This run needs Detector, AOT ONNX (33 MB)")
        ctx.extras["answers"] = ["run"]  # Start anyway: the run downloads them itself
        assert await ensure_models(ctx, manager, {}, kinds=("detector",), action="Detect") is True
        assert ctx.extras["asked"][-1][1].startswith("Detect needs Detector (11 MB)") and fake.calls == []
        ctx.extras["answers"] = ["download"]
        assert await ensure_models(ctx, manager, {}) is True
        assert fake.calls == [("download", "det"), ("download", "aot_onnx")]
        assert set(ctx.extras["manga_models_sheet"].rows) == {"det", "aot_onnx"}
        asked = len(ctx.extras["asked"])
        assert await ensure_models(ctx, manager, {}) is True and len(ctx.extras["asked"]) == asked  # all there
        fake.installed.clear()
        fake.fail = RuntimeError("HTTP 503")
        ctx.extras["answers"] = ["download"]
        assert await ensure_models(ctx, manager, {}) is False  # a failed download stops the run

    asyncio.run(scenario())
    tab = me.EditorTab.__new__(me.EditorTab)
    tab.boxes = []
    assert tab.step_model_kinds("translate") == ("detector", "inpaint") and tab.step_model_kinds("recognize") == (
        "detector",)
    tab.boxes = [{"x": 0}]
    assert tab.step_model_kinds("recognize") == () and tab.step_model_kinds("clean") == ("inpaint",)
    assert tab.step_model_kinds("translate_all") == ("detector", "inpaint") and tab.step_model_kinds("render") == ()


@needs_flet
def test_editor_tab_boxes_steps_sheet_and_ocr(iso, tmp_path, monkeypatch):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.manga import editor as me
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    pages = [write_png(tmp_path / "in" / f"{n}.png", 100, 200) for n in (1, 2)]
    fake_session = FakeEditorSession(pages)
    monkeypatch.setitem(sys.modules, "manga_editor_core",
                        _module("manga_editor_core", MangaEditorSession=lambda *a, **k: fake_session,
                                default_state_file=lambda: str(tmp_path / "state.json")))
    monkeypatch.setattr(me, "image_size", lambda path: (100, 200))
    monkeypatch.setitem(sys.modules, "manga_models", None)  # no download questions here (see ensure_models)

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        jobs = FakeJobs()
        files = FakeFiles(picks=[[str(tmp_path / "ocr.json")]])
        ctx = _ctx(page, {}, jobs=jobs, files=files, output_root=str(iso["Output"]))
        manga = _session(tmp_path, {})
        manga.files.host.selected_files = list(pages)
        manga.loaded = True
        screen = MangaScreen(parse_route("/tools/manga?tab=editor"), ctx, session=manga)
        _mount(page, screen.get_body())
        tab = screen.editor_tab
        await tab.open_page(0)
        assert tab.image_path == pages[0] and fake_session.calls[0] == ("open_page", pages[0])
        assert manga.editor is fake_session and svc.editor_session(manga.editor_token) is fake_session
        tab.geometry.display_w = 50.0  # displayed at half size: scale 0.5
        tab.set_tool("box")
        tab.pan_start(5, 10)
        tab.pan_update(25, 40)
        assert tab.drag["current"] == (25, 40)
        await tab.pan_end()
        assert fake_session.calls[-1] == ("add_box", 10, 20, 40, 60, "rect", False)
        assert len(tab.boxes) == 1 and tab.selected == 0
        tab.set_tool("lasso")
        tab.pan_start(0, 0)
        for point in ((20, 0), (20, 20), (0, 20)):
            tab.pan_update(*point)
        await tab.pan_end()
        assert fake_session.calls[-1] == ("add_box", 0, 0, 40, 40, "polygon", True)
        tab.set_tool("select")
        tab.selected = None  # (the new lasso box's corner handle is within reach of this point)
        tab.pan_start(15, 35)  # inside box 0 only (image 30, 70)
        tab.pan_update(20, 40)
        await tab.pan_end()
        assert fake_session.calls[-1] == ("update_box", 0, 20, 30, 40, 60, False)
        assert tab.tap_at(1, 1) == 1 and tab.tap_at(48, 99) is None
        tab.selected = 0
        await tab.toggle_exclude()
        assert fake_session.calls[-1] == ("set_box_excluded", 0, True) and tab.boxes[0]["exclude_from_clean"]
        sheet = tab.long_press_at(15, 35)
        assert sheet is ctx.extras["manga_box_sheet"] and sheet.index == 0
        # not recognized yet: the OCR text is read-only and Translate this text waits for OCR (desktop flow)
        assert sheet.ocr_field.read_only and sheet.translate_button.disabled and not sheet.ocr_button.disabled
        sheet.translation_field.value = "Hello!"
        sheet.free_text.value = True
        await tab.box_action("save", 0, sheet.changes())
        assert ("edit_box_text", 0, None, "Hello!") in fake_session.calls
        assert ("set_box_free_text", 0, True) in fake_session.calls
        job_id = await tab.box_action("translate", 0, {})
        spec = jobs.specs[-1]
        assert spec.params == {"step": "translate_box", "session": manga.editor_token, "image": os.path.abspath(pages[0]),
                               "index": 0}
        assert tab.busy and tab.step_buttons["detect"].disabled
        assert await tab.run_step("detect") is None and ctx.notes[-1][0] == "Wait for the running step"
        jobs.finish(job_id, result={"manga_step": "translate_box"})
        await _settle()
        assert tab.step_status.value == "Translate this text done" and not tab.busy
        # Detect with boxes asks first
        ctx.extras["answers"] = ["no"]
        assert await tab.run_step("detect") is None
        ctx.extras["answers"] = ["yes"]
        job_id = await tab.run_step("detect")
        assert jobs.specs[-1].params["step"] == "detect"
        jobs.finish(job_id, result={"manga_step": "detect"})
        await _settle()
        # Translate: the rendered page shows through a versioned copy
        job_id = await tab.run_step("translate")
        fake_session.translate(pages[0])
        jobs.finish(job_id, result={"manga_step": "translate"})
        await _settle()
        assert tab.view == "translated" and "_v1_" in os.path.basename(tab.translated_view)
        assert tab.translated_view.startswith(manga.view_cache)
        job_id = await tab.run_step("translate_all")
        assert jobs.specs[-1].params["images"] == [os.path.abspath(p) for p in pages]
        jobs.finish(job_id, result={"manga_step": "translate_all"})
        await _settle()
        # OCR JSON
        path = await tab.export_ocr()
        assert path and path.startswith(str(iso["Output"])) and fake_session.calls[-1][0] == "export_ocr"
        ocr = tmp_path / "ocr.json"
        ocr.write_text("{}", encoding="utf-8")
        job_id = await tab.import_ocr()
        assert jobs.specs[-1].params["step"] == "import_ocr" and jobs.specs[-1].params["path"] == str(ocr)
        jobs.finish(job_id, result={"manga_step": "import_ocr",
                                    "manga_import": {"matched": 1, "files": 2, "translated_regions": 0}})
        await _settle()
        assert tab.step_status.value == "Imported OCR for 1 of 2 pages"
        # ... which the next Start reuses (Files › Run shows it, clearable)
        assert manga.imported_ocr == {"path": str(ocr), "matched": 1, "files": 2}
        files_tab = screen.files_tab
        assert files_tab.imported_row.visible and files_tab._imported_ocr_path() == str(ocr)
        assert svc.batch_spec(manga.files, imported_ocr=files_tab._imported_ocr_path()).params["imported_ocr"] == str(ocr)
        files_tab.clear_imported_ocr()
        assert manga.imported_ocr is None and not files_tab.imported_row.visible
        await tab.step_page(1)
        assert tab.image_path == pages[1] and tab.boxes == []
        assert tab.handle_back() is True and tab.tool == "pan"  # back leaves the edit tool first
        await tab.delete_selected()
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_editor_without_the_core_shows_why(tmp_path, monkeypatch):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    page_path = write_png(tmp_path / "1.png")
    monkeypatch.setitem(sys.modules, "manga_editor_core", None)

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        ctx = _ctx(session.page, {})
        manga = _session(tmp_path, {})
        manga.files.host.selected_files = [page_path]
        screen = MangaScreen(parse_route("/tools/manga?tab=editor"), ctx, session=manga)
        _mount(session.page, screen.get_body())
        tab = screen.editor_tab
        assert await tab.open_page(0) == {}
        assert "manga_editor_core" in (manga.editor_error or "") and ctx.notes
        assert tab.step_reason.content is not None and all(b.disabled for b in tab.step_buttons.values())

    asyncio.run(scenario())


def test_editor_geometry_helpers():
    from glossarion_mobile.ui.tools.manga import editor as me

    g = me.Geometry(image_w=200, image_h=400, display_w=100)
    assert g.scale == 0.5 and g.to_image(10, 20) == (20, 40) and g.to_image(500, -5) == (200, 0)
    assert me.normalize_rect(10, 30, 0, 5) == (0, 5, 10, 25)
    boxes = [{"x": 0, "y": 0, "width": 50, "height": 50}, {"x": 10, "y": 10, "width": 10, "height": 10}]
    assert me.hit_test(boxes, 15, 15) == 1 and me.hit_test(boxes, 40, 40) == 0 and me.hit_test(boxes, 60, 0) is None
    assert me.on_resize_handle(boxes[0], 48, 52, 4) and not me.on_resize_handle(boxes[0], 10, 10, 4)
    assert me.lasso_bounds([(0, 0), (1, 1)]) is None and me.lasso_bounds([(0, 0), (2, 0), (2, 2)]) is None
    bounds, polygon = me.lasso_bounds([(0, 0), (20, 0), (20, 30)])
    assert bounds == (0, 0, 20, 30) and polygon[1] == [20, 0]


def _by_key(control) -> dict:
    """``{key: control}`` for every keyed control under ``control`` (per-build ``-g<n>`` suffix dropped)."""
    found: dict = {}
    stack = [control]
    while stack:
        node = stack.pop()
        key = getattr(node, "key", None)
        if isinstance(key, str) and key:
            found[key.rsplit("-g", 1)[0]] = node
        for attr in ("controls", "content", "trailing", "leading", "title", "subtitle"):
            child = getattr(node, attr, None)
            if isinstance(child, (list, tuple)):
                stack.extend(c for c in child if c is not None and not isinstance(c, (str, int, float, bool)))
            elif child is not None and not isinstance(child, (str, int, float, bool)):
                stack.append(child)
    return found


@pytest.mark.skipif(not _has("settings_schema"), reason="settings_schema not importable")
def test_generic_choice_tiles_see_the_manga_value_rules():
    """All manga settings (schema tiles) refuse the values settings_schema marks unavailable on mobile
    (``SchemaAccess.value_availability`` -> ``is_value_available``), e.g. the Windows-only hard RAM cap."""
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    schema = SchemaAccess(platform="mobile")
    ok, reason = schema.value_availability("manga_settings.advanced.ram_cap_mode", "hard")
    assert not ok and "Windows" in reason
    assert schema.value_availability("manga_settings.advanced.ram_cap_mode", "soft") == (True, None)
    assert SchemaAccess(platform="desktop").value_availability("manga_settings.advanced.ram_cap_mode", "hard")[0]


@needs_flet
@pytest.mark.skipif(not _has("settings_schema"), reason="settings_schema not importable")
def test_settings_tab_builds_every_provider_inpainting_and_rendering_branch(iso, tmp_path, monkeypatch):
    from glossarion_mobile.ui.components.reason_chip import ReasonChip
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.settings.context import SettingsContext
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    monkeypatch.setitem(sys.modules, "manga_models", FakeModels())
    store, schema = _settings_store(tmp_path)

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        settings = SettingsContext(page=page, store=store, schema=schema)
        ctx = _ctx(page, store, settings=settings, output_root=str(iso["Output"]))
        screen = MangaScreen(parse_route("/tools/manga?tab=settings"), ctx, session=_session(tmp_path, {}))
        _mount(page, screen.get_body())
        tab = screen.settings_tab
        tab.did_show()
        # detection: the ONNX export rows of the registry; picking one moves the download row with it
        assert "ms-variant-det.onnx" in _by_key(tab.detection_card)
        tab._select_variant("det.onnx")
        assert store.get(("manga_settings", "ocr", "rtdetr_onnx_variant")) == "det.onnx"
        assert tab.model_rows["detector"].model_id == "det"
        ocr = _by_key(tab.ocr_card)
        # an unset value shows the default the desktop tab runs with (manga_settings_defaults)
        assert ocr["ms-ocr-batch"].value is True and ocr["ms-ocr-batch-size"].value == "5"
        assert tab.select_provider("google") and "ms-google-creds" in _by_key(tab.ocr_card)
        assert tab.select_provider("azure-document-intelligence")
        assert {"ms-docintel-key", "ms-docintel-endpoint"} <= set(_by_key(tab.ocr_card))
        assert tab.select_provider("rapidocr")
        assert tab.select_inpaint("cloud")
        assert {"ms-replicate-key", "ms-inpaint-quality"} <= set(_by_key(tab.inpaint_card))
        assert tab.select_inpaint("local", "custom-image-edit")
        inpaint = _by_key(tab.inpaint_card)
        assert inpaint["ms-edit-batch"].value is True
        shared_test = callable(svc.core_attr("manga_env", "test_custom_image_edit_endpoint"))
        assert inpaint["ms-edit-test"].disabled is (not shared_test)  # the shared desktop test (manga_env)
        assert "ms-edit-keys" in inpaint  # Image keys -> the inpainter key pool
        inpaint["ms-edit-keys"].on_click(None)
        assert ctx.navigated[-1][:2] == ("settings.keys.pool", {"pool": "inpainter"})
        # U9: Model information ⓘ, the mask presets of the desktop dialog (B&W Manga / Colored / Uniform)
        assert {"ms-model-info", "ms-mask-bw_manga", "ms-mask-colored", "ms-mask-uniform"} <= set(inpaint)
        assert tab.show_model_info("aot")
        assert tab.apply_mask_preset("uniform")
        assert store.get(("manga_settings", "mask_dilation")) == 0
        assert store.get(("manga_settings", "use_all_iterations")) is True
        assert store.get(("manga_settings", "empty_bubble_dilation_iterations")) == 2
        assert tab.select_inpaint("local", "anime_onnx") and "inpaint" not in tab.model_rows  # not in FakeModels
        assert tab.select_inpaint("local", "aot_onnx") and tab.model_rows["inpaint"].model_id == "aot_onnx"
        assert "ms-model-import" in _by_key(tab.inpaint_card)  # desktop Browse -> Import model file…
        tab.set("manga_aot_onnx_model_path", str(tmp_path / "mine.onnx"))
        tab.refresh(push_now=False)
        assert "ms-model-clear" in _by_key(tab.inpaint_card)
        tab.clear_model_file("aot_onnx")
        assert store.get("manga_aot_onnx_model_path") == ""
        for mode, key in (("multiplier", "ms-font-mult"), ("fixed", "ms-font-size")):
            tab._set_and_refresh("manga_font_size_mode", mode)
            assert key in _by_key(tab.render_card)
        tab._set_and_refresh("manga_font_size_mode", "auto")
        render = _by_key(tab.render_card)
        assert "ms-font-size" not in render and "ms-font-mult" not in render and "ms-shadow-color-hex" in render
        tab._set_and_refresh("manga_shadow_enabled", False)
        render = _by_key(tab.render_card)
        assert "ms-shadow-color-hex" not in render
        # presets and Reset come from the shared module (enabled with it; a reason chip without it)
        assert render["ms-preset-small"].disabled is (not svc.presets_available())
        assert render["ms-render-reset"].disabled is (not svc.rendering_reset_available())
        has_chip = any(isinstance(c, ReasonChip) for row in tab.render_card.content.controls
                       for c in (getattr(row, "controls", None) or ()))
        assert has_chip is (not (svc.presets_available() and svc.rendering_reset_available()))
        # the Files tab's selection writes are not settings: no re-render for them
        gen = tab._gen
        tab._on_config_changed("manga_selected_files")
        assert tab._gen == gen
        tab._on_config_changed("manga_bg_style")
        assert tab._gen == gen + 1
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_import_model_file_picks_any_file_on_mobile_and_moves_the_inbox_copy(iso, tmp_path):
    """Inpainting › Import model file…: no model extension has an Android MIME type / iOS UTI, so the
    mobile picker gets no filter and the pick is checked afterwards (a wrong file is refused and its
    Inbox copy removed); the model's Inbox copy moves into <root>/models instead of being stored twice.
    The desktop keeps the Browse filter."""
    from glossarion_mobile.ui.tools.manga.settings import SettingsTab

    inbox = tmp_path / "Inbox"
    inbox.mkdir()

    class PickFiles:
        def __init__(self, platform):
            self.platform = platform
            self.calls: list = []
            self.next = ("", False)

        async def pick_files(self, **kwargs):
            self.calls.append(kwargs)
            name, reused = self.next
            path = inbox / name
            if not path.exists():
                path.write_bytes(b"model bytes")
            return [types.SimpleNamespace(path=str(path), name=name, reused=reused)]

    store: dict = {}
    files = PickFiles("android")
    ctx = _ctx(None, store, files=files)
    session = _session(tmp_path, store)
    tab = SettingsTab(ctx, session)
    models = Path(session.root) / "models"

    async def scenario():
        files.next = ("notes.txt", False)
        assert await tab.import_model_file("aot") is None
        assert files.calls[-1]["allowed_extensions"] is None  # the mobile picker shows every file
        assert "not a model file" in ctx.notes[-1][0] and not (inbox / "notes.txt").exists()
        assert not store.get("manga_aot_model_path")
        files.next = ("big.onnx", False)
        path = await tab.import_model_file("aot")
        assert Path(path) == models / "big.onnx" and Path(path).is_file()
        assert not (inbox / "big.onnx").exists() and store["manga_aot_model_path"] == path  # moved, not copied
        (inbox / "kept.pt").write_bytes(b"kept")
        files.next = ("kept.pt", True)  # identical content was already in the Inbox: that file stays
        assert Path(await tab.import_model_file("lama")) == models / "kept.pt" and (inbox / "kept.pt").exists()
        files.platform = "windows"
        files.next = ("desk.safetensors", False)
        assert await tab.import_model_file("aot")
        assert files.calls[-1]["allowed_extensions"] == list(SettingsTab.MODEL_FILE_EXTENSIONS)

    asyncio.run(scenario())


@needs_flet
@pytest.mark.skipif(not _has("PIL") or not _has("safe_image"), reason="Pillow / safe_image not importable")
def test_editor_image_size_reads_headers_through_safe_image(tmp_path, monkeypatch):
    import safe_image

    from glossarion_mobile.ui.tools.manga import editor as me

    # media_model.image_size imports safe_image.open_image when it is called (5f5d2709)
    seen: list = []
    real = safe_image.open_image
    monkeypatch.setattr(safe_image, "open_image", lambda path, *a, **k: seen.append(path) or real(path, *a, **k))
    png = write_png(tmp_path / "size.png", 30, 50)
    assert me.image_size(png) == (30, 50) and seen == [png]
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert me.image_size(str(bad)) == (0, 0) and me.image_size(str(tmp_path / "missing.png")) == (0, 0)


@needs_flet
@pytest.mark.skipif(not (_has("manga_editor_core") and _has("manga_files_core") and _has("PIL")),
                    reason="the shared manga editor core is not importable")
def test_editor_tab_edits_boxes_on_the_real_editor_session(iso, tmp_path, monkeypatch):
    """The editor screen over ``manga_editor_core.MangaEditorSession`` itself: box edits, texts and the
    OCR JSON export need no owner (detect / OCR / translate run as jobs and are faked above)."""
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")  # the state manager runs in-process, as on mobile
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.chdir(tmp_path)
    pages = [write_png(tmp_path / "in" / f"{n}.png", 100, 200) for n in (1, 2)]

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        ctx = _ctx(page, {}, files=FakeFiles(), output_root=str(iso["Output"]))
        manga = _session(tmp_path, {})
        manga.files.host.selected_files = list(pages)
        manga.loaded = True
        screen = MangaScreen(parse_route("/tools/manga?tab=editor"), ctx, session=manga)
        _mount(page, screen.get_body())
        tab = screen.editor_tab
        snap = await tab.open_page(0)
        assert manga.editor is not None, manga.editor_error
        assert snap["image_path"] == os.path.abspath(pages[0]) and tab.boxes == []
        assert (tab.geometry.image_w, tab.geometry.image_h) == (100.0, 200.0)
        assert manga.editor.image_state_manager.state_file_path.startswith(str(iso["data"]))
        tab.geometry.display_w = 50.0  # displayed at half size
        tab.set_tool("box")
        tab.pan_start(5, 10)
        tab.pan_update(25, 40)
        await tab.pan_end()
        assert [(b["x"], b["y"], b["width"], b["height"], b["shape"]) for b in tab.boxes] == [
            (10.0, 20.0, 40.0, 60.0, "rect")]
        tab.set_tool("select")
        tab.selected = None  # (a press next to the corner handle would resize)
        tab.pan_start(15, 35)
        tab.pan_update(20, 40)
        await tab.pan_end()
        assert (tab.boxes[0]["x"], tab.boxes[0]["y"]) == (20.0, 30.0)
        await tab.toggle_exclude()
        assert tab.boxes[0]["exclude_from_clean"] is True
        sheet = tab.long_press_at(25, 45)
        assert sheet is not None and sheet.ocr_field.read_only and sheet.translate_button.disabled
        sheet.translation_field.value = "Hello!"
        sheet.free_text.value = True
        await tab.box_action("save", 0, sheet.changes())
        assert tab.boxes[0]["translation"] == "Hello!" and tab.boxes[0]["free_text"] is True
        path = await tab.export_ocr()
        assert path and path.startswith(str(iso["Output"]))
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        assert document["format"] == "glossarion-manga-ocr"
        assert any(r.get("translated_text") == "Hello!" for r in document["pages"][0]["regions"])
        await tab.step_page(1)
        assert tab.image_path == pages[1] and tab.boxes == []
        await tab.step_page(-1)  # the first page's box and translation come back from the page state
        assert len(tab.boxes) == 1 and tab.boxes[0]["translation"] == "Hello!"
        tab.selected = 0
        assert await tab.delete_selected() and tab.boxes == []
        # U9: Clear boxes (the desktop's Clear Boxes halves, manga_editor_core): confirm, then every box goes
        tab.set_tool("box")
        tab.pan_start(5, 10)
        tab.pan_update(25, 40)
        await tab.pan_end()
        assert len(tab.boxes) == 1
        ctx.extras["answers"] = ["no"]
        assert not await tab.clear_boxes() and len(tab.boxes) == 1
        ctx.extras["answers"] = ["yes"]
        assert await tab.clear_boxes() and tab.boxes == []
        screen.dispose()
        manga.editor.close()

    asyncio.run(scenario())


# ==========================================================================
# What a mobile run starts from: phone defaults, models, output paths, archives, editor gestures
# ==========================================================================


@pytest.fixture
def mobile(iso, monkeypatch):
    """The app's environment contract marks the process as mobile (the iso fixture clears it)."""
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    return iso


class FakeDownloads:
    """``manga_models.download`` stand-in: records (model, cancel), reports progress, can fail."""

    def __init__(self, models) -> None:
        self.models = models
        self.calls: list = []
        self.fail = None

    def __call__(self, model, *, progress=None, cancel=None, **_kwargs):
        spec = self.models.get_spec(model)
        self.calls.append((spec.key, cancel))
        if self.fail is not None:
            raise self.fail
        for done in (0, spec.size // 2, spec.size):
            progress(self.models.DownloadProgress(spec.key, done, spec.size))
        return self.models.model_path(spec)


def _capturing_runner(monkeypatch, seen: dict):
    import copy

    class Capturing(FakeRunner):
        def __init__(self, main_gui, **kwargs):
            super().__init__(main_gui, **kwargs)
            seen["env"] = os.environ.get("OUTPUT_DIRECTORY")
            seen["config"] = copy.deepcopy(main_gui.config)

    FakeRunner.instances = []
    FakeRunner.summary = {"ok": True, "completed": 1, "failed": 0, "total": 1, "outputs": [], "cbz_paths": [],
                          "stopped": False, "error": ""}
    FakeRunner.hold = None
    monkeypatch.setitem(sys.modules, "manga_runner", _module("manga_runner", HeadlessMangaRunner=Capturing,
                                                             MangaRunError=FakeRunError))
    return Capturing


@needs_cores
def test_mobile_batch_runs_with_the_phone_defaults_its_models_and_no_output_override(mobile, tmp_path, monkeypatch):
    import manga_models

    pages = [write_png(tmp_path / "in" / f"{n}.png") for n in (1, 2)]
    seen: dict = {}
    _capturing_runner(monkeypatch, seen)
    downloads = FakeDownloads(manga_models)
    monkeypatch.setattr(manga_models, "download", downloads)
    stored = {"manga_ocr_provider": "azure-document-intelligence", "azure_document_intelligence_key": "k" * 32,
              "azure_document_intelligence_endpoint": "https://di.example/", "output_directory": "C:/desktop/out"}
    owner = types.SimpleNamespace(config=dict(stored))
    params = {"files": pages, "run_files": pages, "output_root": str(mobile["Output"])}
    ctx = FakeCtx(owner, params=params, inputs=pages)
    assert manga_kind.run_batch(ctx)["ok"] is True
    config = seen["config"]
    # the run uses what Settings shows (manga_models phone defaults; the desktop fallbacks otherwise)
    manga = config["manga_settings"]
    assert manga["ocr"]["rtdetr_onnx_variant"] == "detector-v4-s_int8.onnx"
    assert manga["inpainting"]["local_method"] == config["manga_local_inpaint_model"] == "aot_onnx"
    assert manga["advanced"]["hd_strategy_resize_limit"] == 1024 and manga["advanced"]["panel_max_workers"] == 1
    # ... after downloading the models it loads, verified and resumable, with progress and the job's Stop
    assert [key for key, _cancel in downloads.calls] == ["rtdetr_v4_s_int8", "aot_onnx"]
    assert all(cancel == ctx.stop_requested for _key, cancel in downloads.calls)
    labels = [data["label"] for kind, data in ctx.events if kind == "progress"]
    assert "Downloading RT-DETR v4-S INT8 · 0%" in labels and "Downloading AOT ONNX · 100%" in labels
    assert any("Downloading AOT ONNX" in line for line in ctx.logs)
    # no output override: the manga code writes each page next to its source (the job's process
    # state puts OUTPUT_DIRECTORY back); the runner gets no output root of its own either
    assert seen["env"] is None and "output_directory" not in config
    assert FakeRunner.instances[-1].kwargs["output_root"] is None
    # Document Intelligence: Settings' fields also fill the Computer Vision pair the start check reads
    assert (config["azure_vision_key"], config["azure_vision_endpoint"]) == ("k" * 32, "https://di.example/")
    assert any("Azure Document Intelligence" in line for line in ctx.logs)
    assert owner.config is not stored and stored["output_directory"] == "C:/desktop/out"
    # a glossary pass only fetches the detector; a Stop during a download ends the job before the run
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(mobile["Output"]))
    downloads.calls.clear()
    manga_kind.run_batch(FakeCtx(types.SimpleNamespace(config={}), params=dict(params, glossary_only=True)))
    assert [key for key, _cancel in downloads.calls] == ["rtdetr_v4_s_int8"]
    runs = len(FakeRunner.instances)
    downloads.fail = manga_models.DownloadCancelled("detector-v4-s_int8.onnx: download cancelled at 10 bytes")
    ctx = FakeCtx(types.SimpleNamespace(config={}), params=params)
    ctx.request("immediate")
    assert manga_kind.run_batch(ctx) == {"ok": None, "outputs": []} and len(FakeRunner.instances) == runs
    assert any("resumes on the next run" in line for line in ctx.logs)
    # cancelled by anything but the job's Stop (a model row's Cancel stops every download of the
    # model): the job fails with the reason instead of ending "done" with nothing done
    with pytest.raises(JobError, match="The model download was cancelled; start again to resume it"):
        manga_kind.run_batch(FakeCtx(types.SimpleNamespace(config={}), params=params))
    assert len(FakeRunner.instances) == runs
    downloads.fail = manga_models.ModelDownloadError("detector-v4-s_int8.onnx: HTTP 404 Not Found")
    with pytest.raises(JobError, match="could not be downloaded: .*HTTP 404"):
        manga_kind.run_batch(FakeCtx(types.SimpleNamespace(config={}), params=params))
    # an editor step fetches the kinds the editor asked for, then runs
    downloads.fail = None
    downloads.calls.clear()
    session = FakeEditorSession(pages)
    token = svc.register_editor_session(session)
    ctx = FakeCtx(types.SimpleNamespace(config={}),
                  params=svc.step_spec("clean", token, pages[0], extra={"model_kinds": ["inpaint"]}).params)
    assert manga_kind.run_step(ctx)["ok"] is True
    assert [key for key, _cancel in downloads.calls] == ["aot_onnx"] and session.calls[-1][0] == "clean"
    assert "Downloading AOT ONNX · 100%" in ctx.phases
    # desktop (no mobile contract): nothing of the above
    monkeypatch.delenv("GLOSSARION_MOBILE")
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(mobile["Output"]))
    downloads.calls.clear()
    manga_kind.run_batch(FakeCtx(types.SimpleNamespace(config=dict(stored)), params=params))
    assert downloads.calls == [] and seen["env"] == str(mobile["Output"])
    assert "rtdetr_onnx_variant" not in (seen["config"].get("manga_settings") or {}).get("ocr", {})
    assert FakeRunner.instances[-1].kwargs["output_root"] == str(mobile["Output"])


@needs_cores
def test_document_intelligence_runs_with_the_fields_settings_offer(mobile, tmp_path, monkeypatch):
    import copy

    di_only = {"manga_ocr_provider": "azure-document-intelligence", "azure_document_intelligence_key": "k" * 32,
               "azure_document_intelligence_endpoint": "https://di.example/", "model": "gpt-4o-mini",
               "api_key": "sk-test"}
    # the status chip says Ready (the desktop rule: either pair), and the run now agrees
    row = next(r for r in svc.ocr_provider_rows(di_only, mobile=True) if r.value == "azure-document-intelligence")
    assert row.status == "ready" and row.chip == "Ready"
    run = copy.deepcopy(di_only)
    assert svc.borrow_azure_credentials(run) == ["azure_vision_key", "azure_vision_endpoint"]
    assert (run["azure_vision_key"], run["azure_vision_endpoint"]) == ("k" * 32, "https://di.example/")
    # only the Computer Vision pair (the desktop status counts it too), with the entry's placeholder endpoint
    cv = {"ocr_provider": "azure-document-intelligence", "azure_vision_key": "cv",
          "azure_vision_endpoint": svc.AZURE_ENDPOINT_PLACEHOLDER, "azure_document_intelligence_endpoint": "https://di/"}
    assert svc.borrow_azure_credentials(cv) == ["azure_document_intelligence_key", "azure_vision_endpoint"]
    assert cv["azure_document_intelligence_key"] == "cv" and cv["azure_vision_endpoint"] == "https://di/"
    # stored values are never replaced; other providers and the desktop are left alone
    both = dict(di_only, azure_vision_key="cv", azure_vision_endpoint="https://cv/")
    assert svc.borrow_azure_credentials(both) == [] and both["azure_vision_key"] == "cv"
    assert svc.borrow_azure_credentials(dict(di_only, manga_ocr_provider="azure")) == []
    assert svc.borrow_azure_credentials(copy.deepcopy(di_only), mobile=False) == []
    # the shared start check (unchanged; desktop bug 1 in DISCREPANCIES "U8 Manga run env") passes on
    # the prepared snapshot
    headless_owner = pytest.importorskip("headless_owner")
    manga_env = pytest.importorskip("manga_env")
    before = dict(os.environ)
    try:
        with pytest.raises(manga_env.MangaRunEnvError, match="Azure credentials not configured"):
            manga_env.build_manga_run_env(headless_owner.HeadlessOwner(copy.deepcopy(di_only)))
        prepared = copy.deepcopy(di_only)
        assert svc.prepare_run(prepared)
        manga_env.build_manga_run_env(headless_owner.HeadlessOwner(prepared))
    finally:
        os.environ.clear()
        os.environ.update(before)


@needs_cores
def test_mobile_pages_with_the_same_name_keep_their_own_outputs(mobile, tmp_path):
    import job_runner

    fixture = make_series(tmp_path)
    files = svc.MangaFileList({}, temp_root=str(mobile["data"] / "manga" / "cbz"))
    assert files.add_paths([fixture["series"]]) == 6
    ch1, ch2 = (os.path.join(fixture["series"], chapter, "1.png") for chapter in ("ch1", "ch2"))
    # next to the source copy, like a desktop without an output folder: chapters never collide
    assert files.output_path_for(ch1) == os.path.join(fixture["series"], "ch1", "1_translated", "1.png")
    assert files.output_path_for(ch2) == os.path.join(fixture["series"], "ch2", "1_translated", "1.png")
    assert os.environ["OUTPUT_DIRECTORY"] == str(mobile["Output"])  # put back after the lookup
    # the OCR Text folder the job's automatic export uses (the app folder)
    assert files.ocr_dir() == os.path.join(str(mobile["data"]), "OCR Text")
    write_png(os.path.join(fixture["series"], "ch1", "1_translated", "1.png"))
    assert files.existing_outputs() == [os.path.join(fixture["series"], "ch1", "1_translated", "1.png")]
    # Create CBZ packs a folder's pages next to them, named after that folder
    assert files.create_cbz([ch1]) == [os.path.join(fixture["series"], "ch1", "ch1_translated.cbz")]
    # while a job owns the process state the Files tab does not touch os.environ
    hold, release = threading.Event(), threading.Event()

    def job():
        with job_runner.JOB_LOCK:
            hold.set()
            release.wait(5)

    thread = threading.Thread(target=job)
    thread.start()
    try:
        assert hold.wait(5)
        # "not known while the job runs" is not "not translated": the callers retry later
        with pytest.raises(svc.MangaBusy):
            files.output_path_for(ch1)
        with pytest.raises(svc.MangaBusy):
            files.existing_outputs()
        with pytest.raises(svc.MangaBusy):
            files.existing_cbz()
        assert files.ocr_dir() == os.path.join(str(mobile["data"]), "OCR Text")  # the last folder computed
        with pytest.raises(svc.MangaBusy):  # never computed in this list yet: no folder to offer
            svc.MangaFileList({}, temp_root=str(mobile["data"] / "manga" / "cbz")).ocr_dir()
        with pytest.raises(svc.MangaBusy):
            files.create_cbz([ch1])
    finally:
        release.set()
        thread.join(5)
    assert os.environ["OUTPUT_DIRECTORY"] == str(mobile["Output"])
    assert svc.hide_output_override() is True and "OUTPUT_DIRECTORY" not in os.environ


def _zip_of(path: Path, page: str, width: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = write_png(path.parent / "src" / page, width, 30)
    with zipfile.ZipFile(path, "w") as zf:
        zf.write(image, page)
    return str(path)


@needs_cores
def test_a_reimported_archive_with_a_reused_name_shows_its_own_pages(iso, tmp_path):
    files = svc.MangaFileList({}, temp_root=str(iso["data"] / "manga" / "cbz"))
    names = lambda: [os.path.basename(p) for p in files.files]  # noqa: E731
    archive = tmp_path / "Inbox" / "chapter.zip"
    files.add_paths([_zip_of(archive, "red_page.png", 10)])
    first = files.folder_roots
    assert names() == ["red_page.png"] and os.path.basename(first[0]) == "chapter"
    files.clear()
    # the Inbox copy's name is reused by a different archive (the first copy was deleted)
    archive.unlink()
    files.add_paths([_zip_of(archive, "blue_page.png", 20)])
    assert names() == ["blue_page.png"] and files.folder_roots != first
    files.clear()
    assert files.add_paths([str(archive)]) == 1 and names() == ["blue_page.png"]  # the same archive: reused
    # a CBZ under a reused name, through the desktop's own extraction
    files.clear()
    cbz = tmp_path / "Inbox" / "vol.cbz"
    files.add_paths([_zip_of(cbz, "old.png", 10)])
    files.clear()
    cbz.unlink()
    files.add_paths([_zip_of(cbz, "new.png", 20)])
    assert names() == ["new.png"] and list(files.cbz_jobs) == [str(cbz)]


@needs_cores
def test_cbz_pages_survive_a_restart_and_create_cbz_packs_them_back(iso, tmp_path):
    fixture = make_series(tmp_path)
    prefs = FakePrefs()
    saved: dict = {}
    temp_root = str(iso["data"] / "manga" / "cbz")
    files = svc.MangaFileList({}, save=saved.update, temp_root=temp_root, config_source=lambda: dict(saved),
                              prefs=prefs)
    assert files.add_paths([fixture["cbz"]]) == 3
    job = files.cbz_jobs[fixture["cbz"]]
    assert prefs.data[svc.PREFS_CBZ_JOBS] == {fixture["cbz"]: job}
    # the app restarts: a new list over the saved selection and the prefs
    again = svc.MangaFileList({}, temp_root=temp_root, prefs=prefs)
    assert again.load(saved) == 3
    assert again.cbz_jobs == {fixture["cbz"]: job} and {again.cbz_job_for(p) for p in again.files} == {fixture["cbz"]}
    # a run wrote the pages into the archive's own output folder (the moved worker's CBZ routing)
    for page in again.files:
        write_png(os.path.join(job["out_dir"], os.path.basename(page)))
    expected = [os.path.join(job["out_dir"], os.path.basename(p)) for p in again.run_files()[0]]
    assert again.existing_outputs() == expected
    packed = os.path.join(os.path.dirname(fixture["cbz"]), "vol1_translated.cbz")
    assert again.create_cbz() == [packed] and again.existing_cbz() == [packed]
    with zipfile.ZipFile(packed) as archive:
        assert sorted(archive.namelist()) == ["page1.png", "page2.png", "page3.png"]
    # mixed selection: the archive's pages go back into it, the folder's pages into their folder's CBZ
    assert again.add_paths([fixture["series"]]) == 6
    chapter_page = os.path.join(fixture["series"], "ch1", "1.png")
    write_png(again.output_path_for(chapter_page))
    archives = again.create_cbz()
    assert packed in archives and len(archives) == 2
    again.clear()
    assert prefs.data[svc.PREFS_CBZ_JOBS] == {}


@needs_flet
def test_editor_pan_mode_leaves_panning_and_zoom_to_the_viewer(iso, tmp_path, monkeypatch):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.manga import editor as me
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    pages = [write_png(tmp_path / "in" / f"{n}.png", 100, 200) for n in (1, 2)]
    fake_session = FakeEditorSession(pages)
    monkeypatch.setitem(sys.modules, "manga_editor_core",
                        _module("manga_editor_core", MangaEditorSession=lambda *a, **k: fake_session,
                                default_state_file=lambda: str(tmp_path / "state.json")))
    monkeypatch.setattr(me, "image_size", lambda path: (100, 200))
    monkeypatch.setitem(sys.modules, "manga_models", None)

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        ctx = _ctx(page, {}, jobs=FakeJobs(), files=FakeFiles(), output_root=str(iso["Output"]))
        manga = _session(tmp_path, {})
        manga.files.host.selected_files = list(pages)
        manga.loaded = True
        screen = MangaScreen(parse_route("/tools/manga?tab=editor"), ctx, session=manga)
        _mount(page, screen.get_body())
        tab = screen.editor_tab
        await tab.open_page(0)
        viewer, surface = tab.interactive, tab.gestures
        assert tab.tool == "pan" and viewer.pan_enabled and viewer.scale_enabled
        # no pan recognizer over the viewer in Pan mode (it would win the gesture arena and swallow
        # one-finger panning); a long-press still opens the box sheet
        assert (surface.on_pan_start, surface.on_pan_update, surface.on_pan_end, surface.on_tap_down) == (
            None, None, None, None)
        assert surface.on_long_press_start is not None
        # a refresh (box picked, step done) keeps the same viewer, so its zoom stays
        tab.selected = None
        tab.refresh()
        await tab._apply_snapshot(await tab._snapshot())
        assert tab.interactive is viewer and tab.viewer_holder.content.content is viewer
        # an edit tool: a viewer that stays put, under new keys, with the drag handlers on the surface
        tab.set_tool("box")
        assert tab.interactive is not viewer and tab.interactive.key != viewer.key
        assert not tab.interactive.pan_enabled and tab.gestures.on_pan_start is not None
        editing = tab.interactive
        tab.refresh()
        assert tab.interactive is editing
        tab.set_tool("pan")
        assert tab.interactive is not editing and tab.gestures.on_pan_start is None
        panning = tab.interactive
        await tab.step_page(1)  # another page: another viewer
        assert tab.interactive is not panning
        # tablets keep the source side (and its zoom) while the translated side is rebuilt
        ctx.tablet = True
        tab.refresh()
        row = tab.viewer_holder.content
        tab.refresh()
        assert tab.viewer_holder.content is row and row.controls[0].content.content is tab.interactive
        screen.dispose()

    asyncio.run(scenario())


# ==========================================================================
# U8 review, second round: busy lookups, glossary auto-load, model rows, Back, archives, job ends
# ==========================================================================


def _hold_job_lock():
    """Another job (an EPUB translation, say) owns the process state until ``release`` is set."""
    import job_runner

    hold, release = threading.Event(), threading.Event()

    def job():
        with job_runner.JOB_LOCK:
            hold.set()
            release.wait(30)

    thread = threading.Thread(target=job, daemon=True)
    thread.start()
    assert hold.wait(5)
    return release, thread


def _translated_series(tmp_path) -> tuple:
    """Inbox/Series/ch1 with its three pages translated next to their source (a mobile run's layout)."""
    fixture = make_series(tmp_path)
    ch1 = os.path.join(fixture["series"], "ch1")
    pages = [os.path.join(ch1, f"{n}.png") for n in (1, 2, 10)]
    outputs = [write_png(os.path.join(ch1, f"{n}_translated", f"{n}.png")) for n in (1, 2, 10)]
    return ch1, pages, outputs


def _fake_editor(monkeypatch, tmp_path, pages):
    from glossarion_mobile.ui.tools.manga import editor as me

    fake_session = FakeEditorSession(pages)
    monkeypatch.setitem(sys.modules, "manga_editor_core",
                        _module("manga_editor_core", MangaEditorSession=lambda *a, **k: fake_session,
                                default_state_file=lambda: str(tmp_path / "state.json")))
    monkeypatch.setattr(me, "image_size", lambda path: (40, 60))
    return fake_session


@needs_flet
@needs_cores
def test_lookups_a_running_job_refused_run_again_once_a_job_ends(mobile, tmp_path, monkeypatch):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    _ch1, pages, outputs = _translated_series(tmp_path)
    _fake_editor(monkeypatch, tmp_path, pages)
    monkeypatch.setitem(sys.modules, "manga_models", None)
    store = {"manga_selected_files": list(pages)}

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        jobs = FakeJobs()
        ctx = _ctx(page, store, jobs=jobs, files=FakeFiles(), output_root=str(mobile["Output"]))
        manga = _session(tmp_path, store)
        screen = MangaScreen(parse_route("/tools/manga?tab=files"), ctx, session=manga)
        _mount(page, screen.get_body())
        release, thread = _hold_job_lock()
        try:
            screen.did_show()
            await _settle(30)
            files_tab, editor = screen.files_tab, screen.editor_tab
            assert len(manga.files.files) == 3 and manga.last_outputs == []
            # "not known while the job runs" is not "not translated"
            assert files_tab.cbz_button.controls[0].tooltip == "Wait for the running job"
            screen.select_tab("editor")
            await _settle(30)
            assert editor.image_path == pages[0] and editor.translated_view == ""
            assert editor._translated_viewer().content.value.startswith("Wait for the running job")
            assert await editor.open_ocr_files() is None
            assert ctx.notes[-1][0] == "Wait for the running job to finish"
            screen.select_tab("files")
            await _settle(10)
            assert manga.last_outputs == []  # still refused: the retry waits for a job to end
        finally:
            release.set()
            thread.join(5)
        # a job of any kind ends: both lookups run again, without leaving the screen
        job_id = await jobs.submit(JobSpec("translation", "Book"))
        jobs.finish(job_id)
        await _settle(30)
        assert sorted(manga.last_outputs) == sorted(outputs)
        assert files_tab.cbz_button.tooltip is None and not files_tab.cbz_button.disabled
        assert editor.translated_view and editor._translated_viewer().content.content.src == editor.translated_view
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
@needs_cores
def test_settings_shows_the_glossary_a_glossary_pass_generated(mobile, tmp_path, monkeypatch):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    ch1, pages, _outputs = _translated_series(tmp_path)
    monkeypatch.setitem(sys.modules, "manga_models", None)
    store: dict = {}
    # where the job writes it: the moved glossary paths without an output override (job's view)
    csv = Path(ch1) / "Glossary" / "ch1_manga_glossary.csv"

    def status(screen):
        tab = screen.settings_tab
        tab.refresh()
        return next(c.value for c in tab.glossary_card.content.controls
                    if str(getattr(c, "key", "") or "").startswith("ms-glossary-status"))

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        jobs = FakeJobs()
        ctx = _ctx(page, store, jobs=jobs, files=FakeFiles(), output_root=str(mobile["Output"]))
        manga = _session(tmp_path, store)
        screen = MangaScreen(parse_route("/tools/manga?tab=files"), ctx, session=manga)
        _mount(page, screen.get_body())
        screen.did_show()
        await _settle()
        tab = screen.files_tab
        assert await tab.add_paths([ch1]) == 3
        assert status(screen) == "No glossary loaded"
        job_id = await tab.start(glossary_only=True)
        assert jobs.specs[-1].params["glossary_only"] is True
        csv.parent.mkdir(parents=True, exist_ok=True)
        csv.write_text("type,raw_name,translated_name\ncharacter,김철수,Kim Cheolsu\n", encoding="utf-8")
        jobs.finish(job_id, result={"manga_glossary_path": str(csv), "manga_glossary_only": True})
        await _settle(30)
        assert store["manga_generated_glossary_path"] == str(csv)
        assert status(screen) == "Generated: ch1_manga_glossary.csv"
        # ... and a selection change keeps it (the auto-load looks where the job wrote it)
        await tab._mutate(manga.files.toggle_skip, pages[0])
        assert store["manga_generated_glossary_path"] == str(csv) and not manga.files.glossary_stale
        # a selection change while another job owns the process state cannot look there: it is
        # marked stale and runs again once a job ends
        release, thread = _hold_job_lock()
        try:
            await tab._mutate(manga.files.toggle_skip, pages[0])
            await _settle(10)
            assert manga.files.glossary_stale and store["manga_generated_glossary_path"] == ""
        finally:
            release.set()
            thread.join(5)
        other = await jobs.submit(JobSpec("translation", "Book"))
        jobs.finish(other)
        await _settle(30)
        assert not manga.files.glossary_stale and store["manga_generated_glossary_path"] == str(csv)
        assert status(screen) == "Generated: ch1_manga_glossary.csv"
        screen.dispose()

    asyncio.run(scenario())


class SteppedModels(FakeModels):
    """FakeModels whose download reports progress several times (the core reports up to 10/s)."""

    def __init__(self, steps: int = 6) -> None:
        super().__init__()
        self.steps = steps
        self.on_report = None

    def download(self, key, progress=None, cancel=None):
        self.calls.append(("download", key))
        for n in range(1, self.steps + 1):
            report = types.SimpleNamespace(fraction=n / (self.steps + 1), phase="download")
            self.active[key] = report
            progress(report)
            if self.on_report is not None:
                self.on_report()
        self.active.pop(key, None)
        self.installed.add(key)
        return f"/m/{key}"


@needs_flet
@pytest.mark.skipif(not _has("settings_schema"), reason="settings_schema not importable")
def test_a_settings_model_download_reports_into_its_row_without_rebuilding_the_tab(iso, tmp_path, monkeypatch):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.settings.context import SettingsContext
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    fake = SteppedModels()
    monkeypatch.setitem(sys.modules, "manga_models", fake)
    store, schema = _settings_store(tmp_path)

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        ctx = _ctx(page, store, settings=SettingsContext(page=page, store=store, schema=schema),
                   output_root=str(iso["Output"]))
        manga = _session(tmp_path, {})
        screen = MangaScreen(parse_route("/tools/manga?tab=settings"), ctx, session=manga)
        _mount(page, screen.get_body())
        tab = screen.settings_tab
        tab.did_show()
        assert tab.select_inpaint("local", "aot_onnx")
        row = tab.model_rows["inpaint"]
        rebuilds: list = []
        renders: list = []
        on_screen: list = []
        original_refresh, original_render = tab.refresh, row.render
        tab.refresh = lambda push_now=True: (rebuilds.append(row.entry.status), original_refresh(push_now))[1]
        row.render = lambda: (renders.append(row.entry.status), original_render())[1]
        fake.on_report = lambda: on_screen.append(tab.model_rows.get("inpaint") is row
                                                  and row.control in tab.inpaint_card.content.controls)
        entry = await row.download()
        assert entry.status == "ready" and fake.calls == [("download", "aot_onnx")]
        # every progress report re-rendered the row; the tab was rebuilt for the two status changes
        assert renders.count("downloading") >= fake.steps and rebuilds == ["downloading", "ready"]
        # the rebuilds kept the row on screen, so the reports kept landing in it
        assert on_screen == [True] * fake.steps
        assert tab.model_rows["inpaint"] is row and row.control in tab.inpaint_card.content.controls
        # an unrelated rebuild re-reads a kept row's status
        fake.installed.discard("aot_onnx")
        original_refresh()
        assert tab.model_rows["inpaint"] is row and row.entry.status == "missing"
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
@needs_cores
def test_back_follows_the_tab_on_screen(iso, tmp_path):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.base import intercepts_back
    from glossarion_mobile.ui.tools.manga.screen import TABS, MangaScreen

    fixture = make_series(tmp_path)

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        store: dict = {}
        ctx = _ctx(page, store, jobs=FakeJobs(), files=FakeFiles(), output_root=str(iso["Output"]))
        manga = _session(tmp_path, store)
        screen = MangaScreen(parse_route("/tools/manga?tab=files"), ctx, session=manga)
        _mount(page, screen.get_body())
        screen.did_show()
        await _settle()
        files_tab, editor = screen.files_tab, screen.editor_tab
        await files_tab.add_paths([fixture["series"]])
        assert intercepts_back(screen)
        # UI_SPEC §1.6 rule 2: selection mode exits before the View pops
        files_tab._on_row_long_press(manga.files.files[0])
        assert files_tab.selection_mode
        assert screen.handle_back() is True and not files_tab.selection_mode and files_tab.selected == set()
        assert screen.handle_back() is False
        # editor state left from the Editor tab neither consumes Back elsewhere nor changes
        editor.tool, editor.selected = "box", 0
        assert screen.handle_back() is False and editor.tool == "box"
        screen.tabs.selected_index = TABS.index("settings")
        assert screen.handle_back() is False and editor.tool == "box"
        screen.tabs.selected_index = TABS.index("editor")
        assert screen.handle_back() is True and editor.selected is None and editor.tool == "box"
        assert screen.handle_back() is True and editor.tool == "pan"
        assert screen.handle_back() is False
        screen.dispose()

    asyncio.run(scenario())


def _cbz_with(path: Path, pages: dict) -> str:
    """A CBZ at ``path`` with ``{member name: page width}`` (sources outside the archive's folder)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, width in pages.items():
            zf.write(write_png(path.parent.parent / "_src" / path.parent.name / name, width, 30), name)
    return str(path)


@needs_cores
def test_cbz_archives_in_folders_and_zips_get_their_own_extraction(iso, tmp_path):
    files = svc.MangaFileList({}, temp_root=str(iso["data"] / "manga" / "cbz"),
                              folders_root=str(iso["data"] / "manga" / "folders"))
    names = lambda: [os.path.basename(p) for p in files.files]  # noqa: E731
    # a picked folder of volumes, cleared, then another series' folder whose volume reuses the name
    series_a, series_b = tmp_path / "picked" / "SeriesA", tmp_path / "picked" / "SeriesB"
    _cbz_with(series_a / "vol1.cbz", {"a01.png": 10, "a02.png": 10})
    _cbz_with(series_b / "vol1.cbz", {"b01.png": 20})
    assert files.add_paths([str(series_a)]) == 2 and names() == ["a01.png", "a02.png"]
    files.clear()
    assert files.add_paths([str(series_b)]) == 1 and names() == ["b01.png"]
    assert {files.cbz_job_for(p) for p in files.files} == {str(series_b / "vol1.cbz")}
    # a re-imported folder whose volume was replaced by a different archive under the same name
    files.clear()
    _cbz_with(series_a / "vol1.cbz", {"new.png": 30})
    assert files.add_paths([str(series_a)]) == 1 and names() == ["new.png"]
    # two series ZIPs (Android's folder fallback) that each hold a vol1.cbz with a page1.png
    files.clear()
    zips = []
    for name, width in (("ZipA", 11), ("ZipB", 22)):
        cbz = _cbz_with(tmp_path / "build" / name / "vol1.cbz", {"page1.png": width})
        archive = tmp_path / "Inbox" / f"{name}.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w") as zf:
            zf.write(cbz, "vol1.cbz")
        zips.append(str(archive))
    assert files.add_paths([zips[0]]) == 1
    first_page = files.files[0]
    first_bytes = Path(first_page).read_bytes()
    assert files.add_paths([zips[1]]) == 1 and len(files.files) == 2
    assert Path(first_page).read_bytes() == first_bytes  # series A's page untouched by series B
    jobs = files.cbz_jobs
    assert len(jobs) == 2 and len({job["extract_dir"] for job in jobs.values()}) == 2
    assert [files.cbz_job_for(p) for p in files.files] == list(jobs)


@needs_flet
@needs_cores
def test_a_batch_that_ended_while_the_screen_was_closed_is_shown_on_return(mobile, tmp_path, monkeypatch):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    ch1, pages, outputs = _translated_series(tmp_path)
    monkeypatch.setitem(sys.modules, "manga_models", None)
    store: dict = {}
    packed = os.path.join(ch1, "ch1_translated.cbz")  # "Create CBZ at end" of a folder selection

    async def open_screen(ctx, manga):
        ctx.page.views[0].controls.clear()  # every visit gets a View of its own (the last one was popped)
        ctx.page.update()
        screen = MangaScreen(parse_route("/tools/manga?tab=files"), ctx, session=manga)
        _mount(ctx.page, screen.get_body())
        screen.did_show()
        await _settle(30)
        return screen

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        jobs = FakeJobs()
        ctx = _ctx(session.page, store, jobs=jobs, files=FakeFiles(), output_root=str(mobile["Output"]))
        manga = _session(tmp_path, store)
        screen = await open_screen(ctx, manga)
        assert await screen.files_tab.add_paths([ch1]) == 3
        job_id = await screen.files_tab.start()
        screen.dispose()  # the user leaves the tool during the run
        with zipfile.ZipFile(packed, "w") as zf:
            zf.write(outputs[0], "1.png")
        jobs.finish(job_id, result={"manga_outputs": outputs, "manga_completed": 3, "manga_cbz": [packed]})
        screen = await open_screen(ctx, manga)
        tab = screen.files_tab
        assert manga.last_cbz == [packed] and manga.last_outputs == outputs
        assert tab.run_status.value == "Done · 3 translated · CBZ: ch1_translated.cbz"
        assert str(tab.output_list.controls[0].key).startswith("mf-cbz-out-")
        assert manga.batch_end_applied == job_id
        tab.run_status.value = ""
        tab.did_show()  # applied once
        assert tab.run_status.value == ""
        screen.dispose()
        # the app restarts: the run's archive of the folder is found again with its pages
        again = _session(tmp_path, store)
        screen = await open_screen(ctx, again)
        assert again.last_cbz == [packed] and sorted(again.last_outputs) == sorted(outputs)
        screen.dispose()

    asyncio.run(scenario())


def _core_defs(module: str) -> dict:
    """``{"": {top-level name: signature or None}, "<Class>": {member: signature or None}}`` of a
    ``src/`` module, read with ``ast`` (no import: it holds where importing a core is skipped)."""
    import ast

    path = SRC_DIR / f"{module}.py"
    if not path.is_file():
        pytest.skip(f"{module}.py is not in this checkout")
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    def sig(node):
        a = node.args
        return [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs], a.kwarg is not None

    out: dict = {"": {}}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[""][node.name] = sig(node)
        elif isinstance(node, ast.ClassDef):
            members: dict = {}
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members[item.name] = sig(item)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    members[item.target.id] = None
            out[node.name] = members
            out[""].setdefault(node.name, None)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[""].setdefault(target.id, None)
    return out


def _takes(signature, *names, strict: bool = False) -> bool:
    args, var_kwargs = signature
    return all(name in args for name in names) or (var_kwargs and not strict)


def test_bindings_match_the_shared_core_contracts():
    """Every shared-core name and keyword the services, job adapters and screens call exists."""
    editor = _core_defs("manga_editor_core")
    session = editor["MangaEditorSession"]
    assert _takes(session["__init__"], "main_gui", "state_file", "image_paths", "log_callback", "event_callback",
                  strict=True)
    calls = {
        "open_page": ("image_path",), "page_snapshot": ("image_path",), "set_pages": ("image_paths",),
        "detect": ("image_path", "owner"), "clean": ("image_path", "owner"), "recognize": ("image_path", "owner"),
        "translate": ("image_path", "owner"), "translate_all": ("image_paths", "owner"),
        "save_and_update_overlay": ("image_path", "owner"),
        "add_box": ("x", "y", "width", "height", "shape", "polygon"),
        "update_box": ("index", "x", "y", "width", "height", "rerender"), "delete_box": ("index",),
        "set_box_free_text": ("index", "free_text"), "set_box_excluded": ("index", "excluded"),
        "set_box_iterations": ("index", "value"), "edit_box_text": ("index", "ocr_text", "translation"),
        "ocr_box": ("index", "owner"), "translate_box": ("index", "owner"), "clean_box": ("index", "owner"),
        "stop": ("force",), "export_ocr": ("destination", "image_paths", "source_root"),
        "import_ocr": ("path", "image_paths", "owner", "render"), "close": (),
    }
    for name, args in calls.items():
        assert name in session and _takes(session[name], *args, strict=True), name
    assert "current_page" in session and "default_state_file" in editor[""]

    runner = _core_defs("manga_runner")
    headless = runner["HeadlessMangaRunner"]
    assert _takes(headless["__init__"], "main_gui", "host", "files", "image_range", "folder_roots", "split_first_level",
                  "skipped", "cbz_jobs", "cbz_image_to_job", "glossary_only", "progress", "output_root",
                  "image_state_manager", strict=True)
    assert "run" in headless and _takes(headless["request_stop"], "graceful", "force", strict=True)
    assert "MangaRunError" in runner[""]

    files_core = _core_defs("manga_files_core")
    env = _core_defs("manga_env")
    moved = (set(files_core["MangaFilesMixin"]) | set(files_core["MangaHooksMixin"]) | set(env["MangaEnvMixin"])
             | set(env["MangaOcrSessionMixin"]))
    used = {"_add_dropped_manga_paths", "_sort_files", "_toggle_skip_processing_for_path", "_parse_manga_image_range",
            "_manga_range_filtered_files", "_manga_process_groups_for_paths", "_persist_selected_files",
            "_load_persisted_files", "_create_cbz_from_isolated_folders", "_finalize_cbz_jobs",
            "_add_cbz_archive_images", "_skip_key_for_path",
            "_is_manually_skipped_processing_file", "_visible_range_skipped_keys", "_prune_skipped_processing_files",
            "_get_manga_output_path_for_file", "_manga_ocr_output_dir", "_manga_ocr_timestamped_export_filename"}
    assert used <= moved, sorted(used - moved)
    assert {"FileListShim", "_natural_sort_key"} <= set(files_core[""])
    assert {"default_manga_ocr_prompt", "default_full_page_context_prompt", "default_manga_glossary_prompt",
            "default_custom_image_edit_system_prompt"} <= set(env[""])
    assert _takes(env[""]["import_ocr_session"], "state", "path", "files", strict=True)

    models = _core_defs("manga_models")
    assert {"get_spec", "specs", "status", "download", "cancel", "delete", "load", "unload", "loaded", "disk_usage",
            "spec_for_inpaint_method", "spec_for_detector_variant", "missing_models", "format_size",
            "apply_mobile_defaults", "apply_mobile_top_level_defaults", "apply_mobile_run_defaults",
            "MOBILE_DETECTOR_KEY", "KIND_DETECTOR", "DownloadCancelled"} <= set(models[""])
    assert _takes(models[""]["download"], "progress", "cancel", strict=True)
    assert _takes(models[""]["apply_mobile_run_defaults"], "config", "force", strict=True)
    assert {"path", "installed", "partial_bytes", "downloading"} <= set(models["ModelStatus"])
    assert {"fraction", "phase"} <= set(models["DownloadProgress"])
    assert {"key", "kind", "title", "size"} <= set(models["ModelSpec"])

    defaults = _core_defs("manga_settings_defaults")
    assert {"default_manga_settings", "MANGA_TOP_LEVEL_DEFAULTS", "deep_merge"} <= set(defaults[""])
    assert "is_value_available" in _core_defs("settings_schema")[""]


def test_new_sources_parse_on_python_310_and_keep_uniform_line_endings():
    import ast

    files = [APP_DIR / "glossarion_mobile" / "services" / "manga.py",
             APP_DIR / "glossarion_mobile" / "job_kinds" / "manga.py",
             APP_DIR / "glossarion_mobile" / "ui" / "chat" / "quick_chips.py",
             *sorted((APP_DIR / "glossarion_mobile" / "ui" / "tools" / "manga").glob("*.py"))]
    for path in files:
        data = path.read_bytes()
        assert data.count(b"\r\n") in (0, data.count(b"\n")), f"mixed line endings in {path.name}"
        ast.parse(data.decode("utf-8"), filename=str(path), feature_version=(3, 10))
    # services / job adapters stay Flet-free (they run on the job thread and in host tools)
    for path in files[:2]:
        assert "import flet" not in path.read_text(encoding="utf-8")


@needs_flet
@needs_cores
def test_empty_manga_tabs_show_the_faded_halgakos(iso, tmp_path):
    """Owner, 2026-10-09: the Manga translator uses the non-chibi Halgakos as a semi-transparent
    placeholder where pages will appear (Files and Editor tabs with nothing loaded)."""
    import flet as ft

    from glossarion_mobile.ui.components.empty_state import HALGAKOS_FULL
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    def faded(control):
        found = []

        def walk(c):
            if isinstance(c, ft.Image) and c.src == HALGAKOS_FULL:
                found.append(c)
            for child in getattr(c, "controls", None) or []:
                walk(child)
            content = getattr(c, "content", None)
            if isinstance(content, ft.Control):
                walk(content)

        walk(control)
        return found

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        ctx = _ctx(page, {}, jobs=FakeJobs(), files=FakeFiles(picks=[]), output_root=str(iso["Output"]))
        manga = _session(tmp_path, {})
        screen = MangaScreen(parse_route("/tools/manga?tab=files"), ctx, session=manga)
        _mount(page, screen.get_body())
        screen.did_show()
        await _settle()
        for tab in (screen.files_tab, screen.editor_tab):
            images = faded(tab.empty)
            assert len(images) == 1 and 0 < images[0].opacity < 0.5, (tab, images)
        assert screen.files_tab.empty.visible

    asyncio.run(scenario())
