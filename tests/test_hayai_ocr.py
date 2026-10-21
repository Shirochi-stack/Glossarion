"""Hayai OCR (the 'hayai-ocr' manga OCR provider): src/hayai_ocr.py + ocr_manager.HayaiOCRProvider.

Covers the host-side math the ONNX graphs rely on (NaFlex size, 2x2 pixel unshuffle against an
independent re-implementation of the upstream DSCProjector gather, 2D rotary tables, the prefill
mask), the pure-Python byte-level BPE decoder, the greedy KV-cache loop against fake sessions (the
exact feeds of the bixii/hayai-ocr-v2.5-nova-onnx tensor contract), and the provider through
OCRManager (downloads through manga_models, BGR -> RGB, one full-crop result).

The real model (~575 MB) runs only when HAYAI_OCR_MODEL_DIR points at a folder holding the three
registry files (hayai-ocr-vision.onnx, hayai-ocr-decoder.onnx, hayai-ocr-tokenizer.json).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import hayai_ocr as H  # noqa: E402


def _real_dir():
    folder = os.environ.get("HAYAI_OCR_MODEL_DIR", "")
    names = ("hayai-ocr-vision.onnx", "hayai-ocr-decoder.onnx", "hayai-ocr-tokenizer.json")
    return folder if folder and all(os.path.isfile(os.path.join(folder, n)) for n in names) else None


# =========================================================================== preprocessing
@pytest.mark.parametrize("shape", [(268, 160), (102, 568), (512, 512), (40, 900), (900, 30), (5, 5), (16, 16)])
@pytest.mark.parametrize("budget", [256, 384, 512])
def test_naflex_size_keeps_aspect_within_the_patch_budget(shape, budget):
    h, w = shape
    th, tw = H.naflex_size(h, w, budget)
    assert th % 16 == 0 and tw % 16 == 0 and th >= 16 and tw >= 16
    assert (th // 16) * (tw // 16) <= budget
    # as large as the budget allows: one more patch row or column would exceed it (unless clamped)
    assert (th // 16 + 1) * (tw // 16 + 1) > budget
    if min(h, w) >= 64:
        assert abs((th / tw) / (h / w) - 1) < 0.35


def test_naflex_size_square():
    assert H.naflex_size(512, 512, 256) == (256, 256)
    assert H.naflex_size(100, 100, 384) == (304, 304)  # 19 x 19 = 361 patches


def test_patchify_layout_and_normalisation():
    rgb = np.zeros((32, 48, 3), np.uint8)
    rgb[:16, 16:32, 0] = 255   # patch (0, 1): red
    rgb[16:, :16, 2] = 255     # patch (1, 0): blue
    pixels, rows, cols = H.patchify(rgb, 6)
    assert (rows, cols) == (2, 3) and pixels.shape == (6, 768) and pixels.dtype == np.float32
    patch = pixels.reshape(6, 16, 16, 3)  # (dy, dx, channel) inside a patch
    assert np.allclose(patch[1, ..., 0], 1.0) and np.allclose(patch[1, ..., 1:], -1.0)
    assert np.allclose(patch[3, ..., 2], 1.0) and np.allclose(patch[0], -1.0)


def _upstream_unshuffle(features, rows, cols, r=2):
    """DSCProjector.gather_patches + transpose (modeling_hayai.py), by index clamping."""
    d = features.shape[-1]
    out_rows, out_cols = -(-rows // r), -(-cols // r)
    tokens = []
    for p in range(out_rows * out_cols):
        oy, ox = divmod(p, out_cols)
        neighbours = []
        for dy in range(r):
            for dx in range(r):
                sy, sx = min(oy * r + dy, rows - 1), min(ox * r + dx, cols - 1)
                neighbours.append(features[sy * cols + sx])
        g = np.stack(neighbours)            # (r*r, d), k = dy * r + dx
        tokens.append(g.T.reshape(d * r * r))  # channel c * r*r + k
    return np.stack(tokens), out_rows, out_cols


@pytest.mark.parametrize("rows,cols", [(2, 2), (3, 5), (5, 3), (1, 1), (4, 7)])
def test_pixel_unshuffle_matches_the_upstream_gather(rows, cols):
    rng = np.random.default_rng(rows * 10 + cols)
    features = rng.standard_normal((rows * cols, 768)).astype(np.float32)
    got, out_rows, out_cols = H.pixel_unshuffle(features, rows, cols)
    want, want_rows, want_cols = _upstream_unshuffle(features, rows, cols)
    assert (out_rows, out_cols) == (want_rows, want_cols)
    assert got.shape == (out_rows * out_cols, 3072)
    np.testing.assert_array_equal(got, want)


def test_rope_tables():
    freqs = 1.0 / (10000.0 ** (np.arange(0, 32, 2) / 32))
    cos, sin = H.rope_tables(2, 3, 1, 0)
    assert cos.shape == sin.shape == (1, 7, 32) and cos.dtype == np.float32
    # vision token 4 is (y=1, x=1); 5 is (1, 2); the BOS (t=0) has angle 0
    np.testing.assert_allclose(cos[0, 5, :16], np.cos(1 * freqs), atol=1e-6)
    np.testing.assert_allclose(sin[0, 5, 16:], np.sin(2 * freqs), atol=1e-6)
    np.testing.assert_allclose(cos[0, 6], 1.0) and np.testing.assert_allclose(sin[0, 6], 0.0)
    step_cos, step_sin = H.rope_tables(0, 0, 1, 7)
    assert step_cos.shape == (1, 1, 32)
    np.testing.assert_allclose(step_sin[0, 0], np.sin(np.concatenate([7 * freqs, 7 * freqs])), atol=1e-6)


def test_prefill_mask_hides_bos_from_vision_rows():
    mask = H.prefill_mask(3)
    assert mask.shape == (1, 1, 4, 4)
    assert (mask[0, 0, :3, :3] == 0).all() and (mask[0, 0, :3, 3] < -1e8).all() and (mask[0, 0, 3] == 0).all()


# =========================================================================== tokenizer
def _byte_token(text):
    table = H._bytes_to_unicode()
    return "".join(table[b] for b in text.encode("utf-8"))


def _tokenizer_file(tmp_path):
    jp = "日本".encode("utf-8")  # 6 bytes: split mid-character across two tokens
    table = H._bytes_to_unicode()
    vocab = {"[PAD]": 0, "[BOS]": 2, "Hello": 4, _byte_token(" world"): 5,
             "".join(table[b] for b in jp[:4]): 6, "".join(table[b] for b in jp[4:]): 7, _byte_token("한국"): 8}
    data = {"added_tokens": [{"id": 16001, "content": "<bos>", "special": True},
                             {"id": 16002, "content": "<eos>", "special": True},
                             {"id": 0, "content": "[PAD]", "special": True}],
            "model": {"type": "BPE", "vocab": vocab, "merges": []}}
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_byte_level_decoder(tmp_path):
    dec = H.ByteLevelDecoder.from_file(str(_tokenizer_file(tmp_path)))
    assert dec.decode([4, 5]) == "Hello world"
    assert dec.decode([16001, 6, 7, 8, 16002, 0]) == "日本한국"  # specials skipped, split bytes rejoined
    assert dec.decode([999999]) == ""


@pytest.mark.skipif(_real_dir() is None, reason="HAYAI_OCR_MODEL_DIR with the three model files is not set")
def test_byte_level_decoder_matches_tokenizers_on_the_real_vocab():
    tokenizers = pytest.importorskip("tokenizers")
    path = os.path.join(_real_dir(), "hayai-ocr-tokenizer.json")
    ref = tokenizers.Tokenizer.from_file(path)
    dec = H.ByteLevelDecoder.from_file(path)
    for text in ("今日はいい天気ですね", "我们明天一起去图书馆看书吧。", "這個問題比我們想像的還要複雜", "오늘은 정말 좋은 날이에요.",
                 "WHERE ARE YOU GOING?!", "ちょっと待って！ドキドキ…"):
        ids = ref.encode(text).ids
        assert dec.decode(ids) == ref.decode(ids, skip_special_tokens=True), text
    assert dec.decode(list(range(16004))) == ref.decode(list(range(16004)), skip_special_tokens=True)


# =========================================================================== decode loop
class FakeVision:
    def __init__(self):
        self.feeds = []

    def run(self, _outputs, feeds):
        self.feeds.append(feeds)
        n = feeds["pixel_values"].shape[1]
        return [np.arange(n * 768, dtype=np.float32).reshape(1, n, 768) / (n * 768)]


class FakeDecoder:
    """Emits ``script`` then EOS; ``present`` grows like the real graph's KV cache."""

    def __init__(self, script):
        self.script = list(script)
        self.feeds = []

    def run(self, _outputs, feeds):
        self.feeds.append(feeds)
        step = len(self.feeds) - 1
        token = self.script[step] if step < len(self.script) else H.EOS_ID
        logits = np.zeros((1, 16004), np.float32)
        logits[0, token] = 1.0
        new = feeds["vision"].shape[1] + feeds["tokens"].shape[1]
        past = feeds["past"]
        present = np.concatenate([past, np.zeros((24, 1, 2, new, 64), np.float32)], axis=3)
        return [logits, present]


def _engine(tmp_path, script, **kwargs):
    sessions = {}

    def factory(path):
        sessions[os.path.basename(path)] = FakeVision() if "vision" in path else FakeDecoder(script)
        return sessions[os.path.basename(path)]

    engine = H.HayaiOCR("x/vision.onnx", "x/decoder.onnx", str(_tokenizer_file(tmp_path)), session_factory=factory,
                        **kwargs)
    return engine, sessions["vision.onnx"], sessions["decoder.onnx"]


def test_greedy_kv_cache_loop_feeds_the_tensor_contract(tmp_path):
    engine, vision, decoder = _engine(tmp_path, [4, 5], max_patches=64)
    text = engine.recognize(np.full((100, 60, 3), 255, np.uint8))
    assert text == "Hello world"
    (vfeed,) = vision.feeds
    rows, cols = vfeed["p_grid"].tolist()
    assert vfeed["p_grid"].dtype == np.int64 and rows * cols <= 64
    assert vfeed["pixel_values"].shape == (1, rows * cols, 768)
    m = ((rows + 1) // 2) * ((cols + 1) // 2)
    prefill, step1, step2 = decoder.feeds  # BOS -> 4, 4 -> 5, 5 -> EOS
    assert prefill["vision"].shape == (1, m, 3072) and prefill["tokens"].tolist() == [[H.BOS_ID]]
    assert prefill["cos"].shape == (1, m + 1, 32) and prefill["mask"].shape == (1, 1, m + 1, m + 1)
    assert prefill["past"].shape == (24, 1, 2, 0, 64)
    assert step1["vision"].shape == (1, 0, 3072) and step1["tokens"].tolist() == [[4]]
    assert step1["past"].shape == (24, 1, 2, m + 1, 64) and step1["mask"].shape == (1, 1, 1, m + 2)
    assert not step1["mask"].any()
    np.testing.assert_allclose(step1["cos"], H.rope_tables(0, 0, 1, 1)[0])  # text position 1
    assert step2["tokens"].tolist() == [[5]] and step2["past"].shape[3] == m + 2
    np.testing.assert_allclose(step2["sin"], H.rope_tables(0, 0, 1, 2)[1])


def test_max_new_tokens_and_stop(tmp_path):
    engine, _vision, decoder = _engine(tmp_path, [4] * 50, max_new_tokens=3)
    assert engine.generate(np.zeros((20, 20, 3), np.uint8)) == [4, 4, 4]
    assert len(decoder.feeds) == 3  # prefill + 2 steps; the third token needs no further step
    engine, _vision, decoder = _engine(tmp_path, [4] * 50)
    assert engine.generate(np.zeros((20, 20, 3), np.uint8), stop=lambda: True) == [4]
    engine, _vision, _decoder = _engine(tmp_path, [])
    assert engine.recognize(np.zeros((20, 20), np.uint8)) == ""  # grey input, EOS at once
    with pytest.raises(ValueError):
        engine.recognize(np.zeros((0, 5, 3), np.uint8))


# =========================================================================== provider
@pytest.fixture
def provider_env(monkeypatch, tmp_path):
    for key in ("GLOSSARION_MOBILE", "GLOSSARION_NO_PROCESSES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ONNX_CACHE_DIR", str(tmp_path / "onnx"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:  # the desktop CI installs no onnxruntime; the engine below is a fake anyway
        import types
        monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    import manga_models
    import ocr_manager

    calls = {"download": [], "shared": None, "images": []}

    def fake_download(spec, **kwargs):
        calls["download"].append((spec.key, callable(kwargs.get("cancel"))))
        return str(tmp_path / spec.filename)

    class FakeEngine:
        def recognize(self, image, stop=None):
            calls["images"].append(np.array(image))
            return "読めた"

    def fake_shared(*paths, **kwargs):
        calls["shared"] = paths
        return FakeEngine()

    monkeypatch.setattr(manga_models, "download", fake_download)
    monkeypatch.setattr(H, "shared", fake_shared)
    return ocr_manager, calls, tmp_path


def test_provider_downloads_the_registry_files_and_reads_a_crop(provider_env):
    ocr_manager, calls, tmp_path = provider_env
    logs = []
    manager = ocr_manager.OCRManager(log_callback=lambda m, level="info": logs.append(m))
    assert "hayai-ocr" in manager.providers
    assert manager.check_provider_status("hayai-ocr") == {"installed": True, "loaded": False}
    assert manager.get_provider("hayai-ocr").models_downloaded() is False
    assert manager.load_provider("hayai-ocr") is True
    assert [key for key, _ in calls["download"]] == ["hayai_ocr_vision", "hayai_ocr_decoder", "hayai_ocr_tokenizer"]
    assert all(cancel for _, cancel in calls["download"])
    assert calls["shared"] == tuple(str(tmp_path / f) for f in
                                    ("hayai-ocr-vision.onnx", "hayai-ocr-decoder.onnx", "hayai-ocr-tokenizer.json"))
    bgr = np.zeros((30, 50, 3), np.uint8)
    bgr[..., 0] = 255  # blue in BGR
    results = manager.detect_text(bgr, "hayai-ocr", confidence=0.6)
    assert [(r.text, r.bbox, r.confidence) for r in results] == [("読めた", (0, 0, 50, 30), 0.6)]
    assert calls["images"][0][0, 0].tolist() == [0, 0, 255]  # handed over as RGB
    manager.detect_text(np.zeros((30, 50, 4), np.uint8), "hayai-ocr")
    assert calls["images"][1].shape == (30, 50, 3)
    manager.shutdown()
    assert manager.get_provider("hayai-ocr").is_loaded is False


def test_provider_load_failure_is_reported(provider_env, monkeypatch):
    ocr_manager, _calls, _tmp = provider_env
    import manga_models

    def broken(spec, **kwargs):
        raise manga_models.ModelDownloadError("offline")

    monkeypatch.setattr(manga_models, "download", broken)
    manager = ocr_manager.OCRManager()
    assert manager.load_provider("hayai-ocr") is False
    assert "offline" in manager.get_provider("hayai-ocr").last_error
    assert manager.detect_text(np.zeros((10, 10, 3), np.uint8), "hayai-ocr") == []


# =========================================================================== real model (opt-in)
@pytest.mark.skipif(_real_dir() is None, reason="HAYAI_OCR_MODEL_DIR with the three model files is not set")
def test_real_model_reads_rendered_text():
    pytest.importorskip("onnxruntime")
    from PIL import Image, ImageDraw, ImageFont

    folder = _real_dir()
    engine = H.HayaiOCR(*(os.path.join(folder, n) for n in
                          ("hayai-ocr-vision.onnx", "hayai-ocr-decoder.onnx", "hayai-ocr-tokenizer.json")))
    cases = [("WHERE ARE YOU GOING?", ("arial.ttf", "DejaVuSans.ttf"))]
    cases.append(("今日はいい天気ですね", ("YuGothM.ttc", "msgothic.ttc", "NotoSansCJK-Regular.ttc")))
    checked = 0
    for text, fonts in cases:
        font = None
        for name in fonts:
            try:
                font = ImageFont.truetype(name, 40)
                break
            except OSError:
                continue
        if font is None:
            continue
        image = Image.new("RGB", (40 + 42 * len(text), 90), "white")
        ImageDraw.Draw(image).text((20, 20), text, font=font, fill="black")
        assert engine.recognize(image).replace(" ", "") == text.replace(" ", "")
        checked += 1
    if not checked:
        pytest.skip("no font for the rendered text")
