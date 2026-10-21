"""Host tests: the Hayai OCR provider ('hayai-ocr', on-device CJK OCR) in Tools › Manga.

* the OCR provider row is listed on mobile (onnxruntime-only, no PyTorch), its status combines the
  three registry files (vision encoder, decoder, tokenizer) of ``manga_models``;
* the model files are registry specs of kind "ocr" that the Model manager lists and that a run with
  the provider downloads first (``ensure_run_models``), and the Settings tab shows their download rows;
* ``hayai_ocr`` (the shared backend module the provider runs) decodes with a fake onnxruntime session.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hayai_ocr_mobile.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import test_manga_ui as mu  # noqa: E402  (its path setup, helpers and the autouse ``iso`` fixture)
from test_manga_ui import iso  # noqa: E402,F401

from glossarion_mobile.services import manga as svc  # noqa: E402

HAYAI = ["hayai_ocr_vision", "hayai_ocr_decoder", "hayai_ocr_tokenizer"]
needs_models = pytest.mark.skipif(not mu._has("manga_models"), reason="manga_models not importable")


def _entry(key, status, size, progress=None):
    return svc.ModelEntry(key, key, kind="ocr", size=size, status=status, progress=progress)


@needs_models
def test_provider_row_is_listed_on_mobile_with_the_combined_model_status(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    assert svc.ocr_model_ids("hayai-ocr") == HAYAI and svc.ocr_model_ids("rapidocr") == []
    assert svc.value_reason(svc.K_PROVIDER, "hayai-ocr", mobile=True) is None
    assert ("hayai-ocr", "🌏 Hayai OCR (CJK, Local)") in svc.OCR_PROVIDERS
    sizes = {"hayai_ocr_vision": 300, "hayai_ocr_decoder": 200, "hayai_ocr_tokenizer": 100}
    states = {key: _entry(key, "ready", size) for key, size in sizes.items()}

    def row(config=None):
        rows = {r.value: r for r in svc.ocr_provider_rows(config or {}, mobile=True, model_status=states.get)}
        return rows["hayai-ocr"]

    assert (row().status, row().chip, row().disabled) == ("ready", "Ready", False)
    assert "Korean" in row().detail
    states["hayai_ocr_decoder"] = _entry("hayai_ocr_decoder", "missing", 200)
    assert (row().status, row().chip) == ("not_downloaded", "Model not downloaded")
    states["hayai_ocr_decoder"] = _entry("hayai_ocr_decoder", "downloading", 200, progress=0.5)
    states["hayai_ocr_tokenizer"] = _entry("hayai_ocr_tokenizer", "missing", 100)
    downloading = row()
    assert downloading.status == "downloading" and downloading.progress == pytest.approx((300 + 100) / 600)
    assert downloading.chip == "Downloading 67%"
    no_detection = {"manga_settings": {"ocr": {"bubble_detection_enabled": False}}}
    assert "bubble detection" in row(no_detection).detail
    # without a model manager the row is still offered; without onnxruntime it is not in this build
    assert {r.value: r for r in svc.ocr_provider_rows({}, mobile=True)}["hayai-ocr"].status == "ready"
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert "hayai-ocr" not in {r.value for r in svc.ocr_provider_rows({}, mobile=True)}


@needs_models
def test_model_files_are_ocr_entries_a_run_downloads_first(monkeypatch):
    import manga_models

    manager = svc.ModelManager()
    entries = [e for e in manager.entries("ocr") if e.id in HAYAI]  # RapidOCR's files are OCR entries too
    assert [e.id for e in entries] == HAYAI and {e.kind for e in entries} == {"ocr"}
    assert all(e.status == "missing" for e in entries)
    assert not manager.can_load("hayai_ocr_vision")  # opened by the provider when a run starts
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    config = {"manga_ocr_provider": "hayai-ocr"}
    assert [k for k in manager.required(config) if k.startswith("hayai")] == HAYAI
    downloaded = []

    def fake_download(spec, progress=None, cancel=None):
        downloaded.append(spec.key)
        progress(types.SimpleNamespace(percent=100))
        return f"/m/{spec.filename}"

    monkeypatch.setattr(manga_models, "download", fake_download)
    reports = []
    paths = svc.ensure_run_models(config, kinds=("ocr",), on_progress=lambda title, pct: reports.append(title))
    assert downloaded == HAYAI and paths == ["/m/hayai-ocr-vision.onnx", "/m/hayai-ocr-decoder.onnx",
                                             "/m/hayai-ocr-tokenizer.json"]
    assert reports == ["Hayai OCR · vision encoder", "Hayai OCR · text decoder", "Hayai OCR · tokenizer"]
    downloaded.clear()
    svc.ensure_run_models({"manga_ocr_provider": "rapidocr"}, kinds=("ocr",))
    assert not set(downloaded) & set(HAYAI)  # RapidOCR downloads its own reading models, never Hayai's


class _HayaiFakeModels(mu.FakeModels):
    """The registry fake of test_manga_ui plus the Hayai OCR files."""

    def __init__(self):
        super().__init__()
        self.specs_list = self.specs_list + [
            types.SimpleNamespace(key=key, kind="ocr", title=key, size=1024 * 1024, selector="hayai-ocr")
            for key in HAYAI]

    def specs_for_ocr_provider(self, provider):
        return [s for s in self.specs_list if s.kind == "ocr" and s.selector == provider]


@mu.needs_flet
@pytest.mark.skipif(not mu._has("settings_schema"), reason="settings_schema not importable")
def test_settings_tab_shows_the_download_rows_of_the_provider(tmp_path, monkeypatch):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.settings.context import SettingsContext
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    monkeypatch.setitem(sys.modules, "manga_models", _HayaiFakeModels())
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    store, schema = mu._settings_store(tmp_path)

    async def scenario():
        _conn, session = mu._tb()._fake_session("android")
        page = session.page
        settings = SettingsContext(page=page, store=store, schema=schema)
        ctx = mu._ctx(page, store, settings=settings, output_root=str(tmp_path / "Output"))
        screen = MangaScreen(parse_route("/tools/manga?tab=settings"), ctx, session=mu._session(tmp_path, {}))
        mu._mount(page, screen.get_body())
        tab = screen.settings_tab
        tab.did_show()
        assert "hayai-ocr" in tab.provider_tiles
        assert tab.select_provider("hayai-ocr")
        assert store.get("manga_ocr_provider") == "hayai-ocr"
        assert {slot: row.model_id for slot, row in tab.model_rows.items() if slot.startswith("ocr-")} == {
            f"ocr-{key}": key for key in HAYAI}
        shown = tab.ocr_card.content.controls
        assert all(tab.model_rows[f"ocr-{key}"].control in shown for key in HAYAI)
        screen.dispose()

    asyncio.run(scenario())


@pytest.mark.skipif(not (mu._has("hayai_ocr") and mu._has("numpy") and mu._has("PIL")),
                    reason="hayai_ocr / numpy / Pillow not importable")
def test_shared_recognizer_decodes_with_a_fake_onnx_session(tmp_path):
    import numpy as np

    hayai_ocr = svc.core("hayai_ocr")
    assert hayai_ocr is not None
    table = hayai_ocr._bytes_to_unicode()
    vocab = {"".join(table[b] for b in "오늘".encode("utf-8")): 4, "".join(table[b] for b in "は晴れ".encode("utf-8")): 5}
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(json.dumps({"model": {"vocab": vocab},
                                     "added_tokens": [{"id": 16001, "content": "<bos>", "special": True},
                                                      {"id": 16002, "content": "<eos>", "special": True}]},
                                    ensure_ascii=False), encoding="utf-8")
    script = [4, 5, hayai_ocr.EOS_ID]
    feeds = []

    class Session:
        def __init__(self, path):
            self.vision = "vision" in path

        def run(self, _outputs, feed):
            feeds.append(feed)
            if self.vision:
                return [np.zeros((1, feed["pixel_values"].shape[1], 768), np.float32)]
            logits = np.zeros((1, 16004), np.float32)
            logits[0, script[len(feeds) - 2]] = 1.0
            grow = feed["vision"].shape[1] + 1
            return [logits, np.zeros((24, 1, 2, feed["past"].shape[3] + grow, 64), np.float32)]

    engine = hayai_ocr.HayaiOCR("vision.onnx", "decoder.onnx", str(tokenizer), session_factory=Session)
    assert engine.recognize(np.full((120, 48, 3), 255, np.uint8)) == "오늘は晴れ"
    assert feeds[1]["tokens"].tolist() == [[hayai_ocr.BOS_ID]] and feeds[2]["tokens"].tolist() == [[4]]
