"""Translate's fallback inpainter load (manga_editor_core._load_inpainter_into_pool) must load the
endpoint-backed custom image edit in-process, like MangaTranslator's shared getter: a worker
process never receives the Image Gen/Edit key model, so its load fails ("no configured model")."""
from types import SimpleNamespace

import pytest

import local_inpainter
import manga_editor_core
from manga_translator import MangaTranslator


class FakeInpainter:
    made = []

    def __init__(self, enable_worker_process=True, **kwargs):
        self.enable_worker_process = enable_worker_process
        self.config = {}
        FakeInpainter.made.append(self)

    def load_model(self, method, model_path, force_reload=False):
        self.model_loaded = True
        return True


@pytest.fixture(autouse=True)
def fake_inpainter(monkeypatch):
    FakeInpainter.made.clear()
    monkeypatch.setattr(local_inpainter, "LocalInpainter", FakeInpainter)
    monkeypatch.setattr(MangaTranslator, "_inpaint_pool", {})


def editor():
    config = {"inpainter_keys": [{"model": "authgem-vertex/gemini-nano-banana-2.1", "enabled": True}],
              "use_inpainter_keys": True}
    return SimpleNamespace(main_gui=SimpleNamespace(config=config))


def test_custom_image_edit_loads_in_process_with_the_key_pool_config():
    key = ("custom-image-edit", "__default_image_edit_endpoint__")
    inp = manga_editor_core._load_inpainter_into_pool(editor(), "custom-image-edit", "", key, True)
    assert inp is FakeInpainter.made[0]
    assert inp.enable_worker_process is False
    assert inp.config["inpainter_keys"][0]["model"] == "authgem-vertex/gemini-nano-banana-2.1"
    assert MangaTranslator._inpaint_pool[key]["spares"] == [inp]


def test_local_model_still_uses_a_worker_process():
    key = ("anime_onnx", "/models/lama.onnx")
    inp = manga_editor_core._load_inpainter_into_pool(editor(), "anime_onnx", "/models/lama.onnx", key, False)
    assert inp.enable_worker_process is True
