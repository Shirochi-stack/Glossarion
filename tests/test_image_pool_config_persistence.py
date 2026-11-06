"""The Image gen/edit key pool stays on and is used (2026-11-06 owner report).

* A bubble detector / local inpainter model load writes only the keys it owns into config.json.
  Before, each wrote back the whole config.json copy it read at creation, which undid every
  setting the GUI saved since (``use_inpainter_keys`` turned off, ``active_profile`` went back).
* ``key_pools.export_key_pool_env`` exports the Image gen/edit and Audio/TTS pools like the
  other dedicated pools.
* "Generate from prompt" (no input file) sets up the Image gen/edit pool and sends the image
  request with a context unified_api_client routes to it, so the pool key is used, not the main key.

Every config file is a temporary one; nothing is sent (the provider call is stubbed at
``UnifiedClient._get_response_once`` and sockets are blocked).

Run (repository root, PYTHONPATH=src)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/test_image_pool_config_persistence.py
"""
import json
import os
import socket
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import app_paths  # noqa: E402
import config_store  # noqa: E402

POOL_KEY = "sk-local-test-key-0001"
MAIN_KEY = "sk-local-test-main-0000"
POOL_ENV = ("USE_INPAINTER_KEYS", "INPAINTER_API_KEYS", "USE_TTS_KEYS", "TTS_API_KEYS")


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "cfg" / "config.json"
    path.parent.mkdir()
    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(path))
    monkeypatch.setattr(config_store, "READ_RETRY_SECONDS", 0)
    monkeypatch.setattr(app_paths, "_WRITE_PROTECTED_FILES", set())
    # no API keys here: skip the encryption key file (it lives next to the code)
    import api_key_encryption
    monkeypatch.setattr(api_key_encryption, "encrypt_config", dict)
    # settings the GUI had saved before the model owner was created (the stale copy it keeps)
    path.write_text(json.dumps({"use_inpainter_keys": False, "active_profile": "TEST",
                                "last_update_check_time": 1.0, "model": "gpt-4o-mini"}), encoding="utf-8")
    return path


def _gui_saves(config_path):
    """The GUI's save path (save_config's final write): the owner turns the pool on, picks a profile."""
    current = config_store.load_config(str(config_path), decrypt=False)
    current.update({"use_inpainter_keys": True, "active_profile": "Manga_JP", "last_update_check_time": 2.0})
    config_store.save_config_file(current, str(config_path), backup=False)


def _on_disk(config_path):
    return json.loads(config_path.read_text(encoding="utf-8"))


def _assert_gui_settings_kept(config_path):
    data = _on_disk(config_path)
    assert data["use_inpainter_keys"] is True
    assert data["active_profile"] == "Manga_JP"
    assert data["last_update_check_time"] == 2.0
    return data


# ---------------------------------------------------------------------------
# 1. model loads never write a stale config copy back
# ---------------------------------------------------------------------------

def test_bubble_detector_model_load_keeps_the_settings_the_gui_saved(config_path, tmp_path, monkeypatch):
    bubble_detector = pytest.importorskip("bubble_detector")
    for name in ("CUDA_LAUNCH_BLOCKING", "TORCH_USE_CUDA_DSA"):  # set by __init__; restored afterwards
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BUBBLE_CACHE_DIR", str(tmp_path / "bubble_cache"))

    class FakeYolo:
        def __init__(self, path):
            self.path = path

    monkeypatch.setattr(bubble_detector, "YOLO_AVAILABLE", True)
    monkeypatch.setattr(bubble_detector, "YOLO", FakeYolo)
    model = tmp_path / "comic-bubbles.pt"
    model.write_bytes(b"weights")

    detector = bubble_detector.BubbleDetector(config_path=str(config_path))
    _gui_saves(config_path)
    assert detector.load_model(str(model)) is True

    data = _assert_gui_settings_kept(config_path)
    assert data["last_model_path"] == str(model)
    assert data["model_type"] == "yolo"
    assert detector.config["last_model_path"] == str(model)


def test_local_inpainter_model_load_keeps_the_settings_the_gui_saved(config_path, tmp_path, monkeypatch):
    local_inpainter = pytest.importorskip("local_inpainter")
    monkeypatch.setattr(local_inpainter, "ONNX_CACHE_DIR", str(tmp_path / "onnx_cache"))
    monkeypatch.setattr(local_inpainter, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(local_inpainter, "BUBBLE_DETECTOR_AVAILABLE", True)

    inpainter = local_inpainter.LocalInpainter(config_path=str(config_path), enable_worker_process=False)
    inpainter.bubble_detector = types.SimpleNamespace(load_model=lambda path: True)
    _gui_saves(config_path)
    assert inpainter.load_bubble_model("models/comic-bubbles.onnx") is True

    data = _assert_gui_settings_kept(config_path)
    assert data["bubble_model_path"] == "models/comic-bubbles.onnx"
    assert inpainter.config["bubble_model_path"] == "models/comic-bubbles.onnx"


def test_an_inpainter_holding_the_gui_config_writes_only_its_model_path(config_path, tmp_path, monkeypatch):
    """manga_editor_core copies the GUI's in-memory config (decrypted keys) into the inpainter."""
    local_inpainter = pytest.importorskip("local_inpainter")
    monkeypatch.setattr(local_inpainter, "ONNX_CACHE_DIR", str(tmp_path / "onnx_cache"))
    monkeypatch.setattr(local_inpainter, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(local_inpainter, "BUBBLE_DETECTOR_AVAILABLE", True)

    inpainter = local_inpainter.LocalInpainter(config_path=str(config_path), enable_worker_process=False)
    inpainter.config.update({"inpainter_keys": [{"api_key": POOL_KEY, "model": "gpt-image-1"}],
                             "use_inpainter_keys": False, "active_profile": "TEST"})
    inpainter.bubble_detector = types.SimpleNamespace(load_model=lambda path: True)
    _gui_saves(config_path)
    assert inpainter.load_bubble_model("models/comic-bubbles.onnx") is True

    data = _assert_gui_settings_kept(config_path)
    assert "inpainter_keys" not in data
    assert POOL_KEY not in config_path.read_text(encoding="utf-8")


def test_update_config_file_reads_and_writes_under_the_config_write_lock(config_path, monkeypatch):
    """A GUI save from another thread cannot land between the read and the write."""
    held = []
    real_load = config_store.load_config_retrying

    def load(*args, **kwargs):
        held.append(app_paths._ATOMIC_WRITE_LOCK._is_owned())
        return real_load(*args, **kwargs)

    monkeypatch.setattr(config_store, "load_config_retrying", load)
    config_store.update_config_file({"bubble_model_path": "m.pt"}, str(config_path))
    assert held == [True]
    assert _on_disk(config_path)["bubble_model_path"] == "m.pt"


# ---------------------------------------------------------------------------
# 2. export_key_pool_env exports the Image gen/edit and Audio/TTS pools
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("enabled", [True, False])
def test_export_key_pool_env_exports_the_image_and_tts_pools(enabled, monkeypatch):
    import key_pools

    for name in POOL_ENV:
        monkeypatch.delenv(name, raising=False)
    image_keys = [{"api_key": POOL_KEY, "model": "gpt-image-1"}]
    tts_keys = [{"api_key": "sk-local-test-key-0002", "model": "tts-1"}]
    key_pools.export_key_pool_env({"use_inpainter_keys": enabled, "inpainter_keys": image_keys,
                                   "use_tts_keys": enabled, "tts_keys": tts_keys})
    flag = "1" if enabled else "0"
    assert os.environ["USE_INPAINTER_KEYS"] == flag
    assert os.environ["USE_TTS_KEYS"] == flag
    assert json.loads(os.environ["INPAINTER_API_KEYS"]) == image_keys
    assert json.loads(os.environ["TTS_API_KEYS"]) == tts_keys


def test_export_key_pool_env_names_match_the_pool_specs():
    """The names unified_api_client reads (instance_plan.POOLS / key_pool_service.POOL_SPECS)."""
    from instance_plan import POOLS

    assert POOLS["inpainter"][2:] == (("USE_INPAINTER_KEYS",), ("INPAINTER_API_KEYS",))
    assert POOLS["tts"][2:] == (("USE_TTS_KEYS",), ("TTS_API_KEYS",))


# ---------------------------------------------------------------------------
# 3. Generate from prompt uses the Image gen/edit pool
# ---------------------------------------------------------------------------

@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("network blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def unified_client(monkeypatch, tmp_path, no_network):
    unified_api_client = pytest.importorskip("unified_api_client")
    UnifiedClient = unified_api_client.UnifiedClient
    sent = []

    def fake_response_once(self, messages, temperature, max_tokens, max_completion_tokens, response_name,
                           request_id=None):
        sent.append({"api_key": self.api_key, "model": self.model})
        return unified_api_client.UnifiedResponse(content="generated text", finish_reason="stop")

    monkeypatch.setattr(UnifiedClient, "_get_response_once", fake_response_once)
    for name in POOL_ENV + ("ENABLE_IMAGE_OUTPUT_MODE", "ENABLE_VIDEO_OUTPUT_MODE", "NANOGPT_VIDEO_DURATION",
                            "NANOGPT_VIDEO_RESOLUTION", "OPENAI_API_KEY", "USE_MULTI_KEYS", "USE_MULTI_API_KEYS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GLOSSARION_DATA_DIR", str(tmp_path / "data"))  # Generated_Media lands here
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    monkeypatch.chdir(tmp_path)  # Payloads/
    yield UnifiedClient, sent
    UnifiedClient.clear_in_memory_inpainter_keys()
    UnifiedClient.clear_in_memory_tts_keys()


def _generative_owner(config, **attrs):
    from image_job import ImageJobMixin
    from run_env import RunEnvMixin

    class Owner(RunEnvMixin, ImageJobMixin):
        def __init__(self):
            self.config = config
            self.model_var = config["model"]
            self.api_key_var = MAIN_KEY
            self.max_output_tokens = 256
            self.stop_requested = False
            self._generative_prompt_override = "A red apple on a wooden table"
            self.logs = []
            for key, value in attrs.items():
                setattr(self, key, value)

        def append_log(self, message):
            self.logs.append(str(message))

    return Owner()


def _pool_config(model, use_pool=True):
    return {"model": model, "use_inpainter_keys": use_pool,
            "inpainter_keys": [{"api_key": POOL_KEY, "model": model, "enabled": True}],
            "force_key_rotation": True, "rotation_frequency": 1}


@pytest.mark.parametrize("model, attrs", [
    ("gpt-image-1", {}),  # an image generation model
    ("gpt-4o-mini", {"enable_image_output_mode_var": True, "enable_image_translation_var": True}),  # image output mode
])
def test_generate_from_prompt_sends_with_the_image_pool_key(unified_client, model, attrs):
    _UnifiedClient, sent = unified_client
    owner = _generative_owner(_pool_config(model), **attrs)
    assert owner._run_generative_prompt_mode() is True, owner.logs
    assert [s["api_key"] for s in sent] == [POOL_KEY], owner.logs
    assert os.environ["USE_INPAINTER_KEYS"] == "1"


def test_generate_from_prompt_with_the_pool_off_uses_the_main_key(unified_client):
    _UnifiedClient, sent = unified_client
    owner = _generative_owner(_pool_config("gpt-image-1", use_pool=False))
    assert owner._run_generative_prompt_mode() is True, owner.logs
    assert [s["api_key"] for s in sent] == [MAIN_KEY], owner.logs
    assert os.environ["USE_INPAINTER_KEYS"] == "0"


def test_input_image_and_prompt_only_runs_share_one_pool_setup():
    import ast
    import inspect

    import image_job

    tree = ast.parse(inspect.getsource(image_job))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ImageJobMixin")
    methods = {n.name: ast.unparse(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
    for name in ("_process_image_file", "_run_generative_prompt_mode"):
        assert "self._configure_media_output_key_pools()" in methods[name]
        assert "set_in_memory_inpainter_keys" not in methods[name]
    assert methods["_configure_media_output_key_pools"].count("set_in_memory_inpainter_keys") == 1
