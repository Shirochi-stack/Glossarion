"""Hayai OCR v2.5 Nova on ONNX Runtime (no PyTorch): the "hayai-ocr" manga OCR provider.

Hayai OCR (JustANormalTinkerer/hayai-ocr-v2.5-nova, Apache-2.0, ~150M parameters) reads a
whole text-region crop (a speech bubble, a caption, a sound effect) of Japanese (horizontal
and vertical), Chinese, Korean or English in one pass: a SigLIP2 NaFlex vision encoder and a
12-layer causal decoder with a 16,004-token byte-level BPE vocabulary.

The graphs are the ONNX export of bixii/hayai-ocr-v2.5-nova-onnx (fp32, opset 18; its model
card documents the tensor contract this module follows):

    vision   in  p_grid        int64  [2]                 patch rows, patch columns
                 pixel_values  float32[1, N, 768]         16x16x3 patches, RGB, (v/255 - 0.5) / 0.5
             out features      float32[1, N, 768]
    decoder  in  vision        float32[1, M, 3072]        2x2 pixel-unshuffled features; M = 0 on a step
                 tokens        int64  [1, T]              [BOS] on prefill, then the last token
                 cos, sin      float32[1, M+T, 32]        2D mRoPE tables
                 mask          float32[1, 1, M+T, L+M+T]  additive; on prefill vision rows may not see BOS
                 past          float32[24, 1, 2, L, 64]   key-value cache, L = 0 on prefill
             out logits        float32[1, 16004]          for the last position
                 present       float32[24, 1, 2, L+M+T, 64]

Host side (numpy only): the NaFlex resize (aspect ratio kept, at most ``max_patches`` 16x16
patches, the binary search of transformers' Siglip2ImageProcessor), the replicate pad + 2x2
pixel unshuffle of the DSCProjector, the 2D rotary tables and a greedy decode loop with the
KV cache. The tokenizer is decoded by a small pure-Python byte-level BPE decoder (the mobile
build has no ``tokenizers`` package; only decoding is needed).

The model files are downloaded on demand by ``manga_models`` (kind "ocr").
"""

from __future__ import annotations

import json
import math
import os
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PATCH_SIZE",
    "DEFAULT_MAX_PATCHES",
    "BOS_ID",
    "EOS_ID",
    "PAD_ID",
    "ByteLevelDecoder",
    "HayaiOCR",
    "naflex_size",
    "patchify",
    "pixel_unshuffle",
    "rope_tables",
    "prefill_mask",
]

PATCH_SIZE = 16
D_VISION = 768
ROPE_AXIS = 32          # head dim 64 -> 32 rotary angles: 16 for y (or t), 16 for x (or t)
ROPE_THETA = 10000.0
N_KV = 24               # 12 layers x (k, v)
KV_HEADS = 2
D_HEAD = 64
#: The model card's default patch budget (256 = throughput, 384 = balanced, 512 = quality).
DEFAULT_MAX_PATCHES = 384
DEFAULT_MAX_NEW_TOKENS = 128
BOS_ID = 16001
EOS_ID = 16002
PAD_ID = 16000
_NEG = np.float32(-1e9)


# =========================================================================== tokenizer
def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2's byte -> printable character table (what the ByteLevel pre-tokenizer encodes with)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class ByteLevelDecoder:
    """Decodes token ids of a Hugging Face ``tokenizers`` byte-level BPE ``tokenizer.json``.

    Special (added) tokens are skipped, like ``decode(..., skip_special_tokens=True)``."""

    def __init__(self, vocab: Dict[str, int], special_ids: Iterable[int] = ()) -> None:
        self.id_to_token: Dict[int, str] = {int(i): str(t) for t, i in vocab.items()}
        self.special_ids = frozenset(int(i) for i in special_ids)
        self._byte_of = {ch: b for b, ch in _bytes_to_unicode().items()}

    @classmethod
    def from_file(cls, path: str) -> "ByteLevelDecoder":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        model = data.get("model") or {}
        vocab = dict(model.get("vocab") or {})
        special = []
        for added in data.get("added_tokens") or ():
            vocab.setdefault(str(added.get("content")), int(added.get("id")))
            if added.get("special", True):
                special.append(int(added.get("id")))
        return cls(vocab, special)

    def decode(self, ids: Sequence[int]) -> str:
        out = bytearray()
        for i in ids:
            i = int(i)
            if i in self.special_ids:
                continue
            token = self.id_to_token.get(i)
            if token is None:
                continue
            for ch in token:
                b = self._byte_of.get(ch)
                if b is None:
                    out += ch.encode("utf-8")
                else:
                    out.append(b)
        return out.decode("utf-8", errors="replace")


# =========================================================================== preprocessing
def naflex_size(height: int, width: int, max_patches: int = DEFAULT_MAX_PATCHES,
                patch: int = PATCH_SIZE, eps: float = 1e-5) -> Tuple[int, int]:
    """The (height, width) a crop is resized to: aspect ratio kept, both sides multiples of
    ``patch``, at most ``max_patches`` patches (Siglip2ImageProcessor's
    ``get_image_size_for_max_num_patches``)."""

    def scaled(scale: float, size: int) -> int:
        return int(max(patch, math.ceil(size * scale / patch) * patch))

    lo, hi = eps / 10, 100.0
    while hi - lo >= eps:
        mid = (lo + hi) / 2
        if (scaled(mid, height) // patch) * (scaled(mid, width) // patch) <= max_patches:
            lo = mid
        else:
            hi = mid
    return scaled(lo, height), scaled(lo, width)


def _to_rgb_array(image: Any) -> np.ndarray:
    """A PIL image or an HxWx3 uint8 RGB array (HxW grey, HxWx4 RGBA accepted) as RGB uint8."""
    if hasattr(image, "convert") and hasattr(image, "size"):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def patchify(rgb: np.ndarray, max_patches: int = DEFAULT_MAX_PATCHES) -> Tuple[np.ndarray, int, int]:
    """NaFlex resize + normalisation + 16x16 patches: ``(pixel_values [N, 768], rows, cols)``.

    Patch features are ordered (dy, dx, channel), patches row-major (SigLIP2's
    ``convert_image_to_patches``)."""
    from PIL import Image

    h, w = rgb.shape[:2]
    th, tw = naflex_size(h, w, max_patches)
    if (th, tw) != (h, w):
        resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
        rgb = np.asarray(Image.fromarray(rgb).resize((tw, th), resample), dtype=np.uint8)
    arr = rgb.astype(np.float32) * np.float32(2.0 / 255.0) - np.float32(1.0)
    rows, cols = th // PATCH_SIZE, tw // PATCH_SIZE
    patches = arr.reshape(rows, PATCH_SIZE, cols, PATCH_SIZE, 3).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(patches.reshape(rows * cols, PATCH_SIZE * PATCH_SIZE * 3)), rows, cols


def pixel_unshuffle(features: np.ndarray, rows: int, cols: int) -> Tuple[np.ndarray, int, int]:
    """DSCProjector's reshape -> replicate pad to even sides -> 2x2 pixel unshuffle.

    ``features`` [rows*cols, 768] -> ``([M, 3072], rows', cols')`` with channel ``c * 4 + dy * 2 + dx``."""
    grid = features.reshape(rows, cols, -1)
    d = grid.shape[-1]
    out_rows, out_cols = (rows + 1) // 2, (cols + 1) // 2
    pad_r, pad_c = out_rows * 2 - rows, out_cols * 2 - cols
    if pad_r or pad_c:
        grid = np.pad(grid, ((0, pad_r), (0, pad_c), (0, 0)), mode="edge")
    tokens = grid.reshape(out_rows, 2, out_cols, 2, d).transpose(0, 2, 4, 1, 3)  # (oy, ox, c, dy, dx)
    return np.ascontiguousarray(tokens.reshape(out_rows * out_cols, d * 4), dtype=np.float32), out_rows, out_cols


def _inv_freq() -> np.ndarray:
    return (1.0 / (ROPE_THETA ** (np.arange(0, ROPE_AXIS, 2, dtype=np.float64) / ROPE_AXIS))).astype(np.float32)


def rope_tables(rows: int, cols: int, n_text: int = 1, text_start: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """cos, sin ``[1, rows*cols + n_text, 32]``: a vision token at grid (y, x) uses ``[y*f, x*f]``,
    text position t uses ``[t*f, t*f]`` (BOS is t = 0). ``rows * cols`` may be 0 (a decode step)."""
    freqs = _inv_freq()
    parts = []
    if rows and cols:
        y = np.repeat(np.arange(rows, dtype=np.float32), cols)
        x = np.tile(np.arange(cols, dtype=np.float32), rows)
        parts.append(np.concatenate([np.outer(y, freqs), np.outer(x, freqs)], axis=-1))
    if n_text:
        t = np.arange(text_start, text_start + n_text, dtype=np.float32)
        angle = np.outer(t, freqs)
        parts.append(np.concatenate([angle, angle], axis=-1))
    ang = np.concatenate(parts, axis=0)[None].astype(np.float32)
    return np.cos(ang), np.sin(ang)


def prefill_mask(m_vision: int) -> np.ndarray:
    """The additive prefill mask ``[1, 1, M+1, M+1]``: vision tokens see each other but not BOS,
    BOS sees everything."""
    mask = np.zeros((1, 1, m_vision + 1, m_vision + 1), dtype=np.float32)
    mask[0, 0, :m_vision, m_vision] = _NEG
    return mask


# =========================================================================== runtime
SessionFactory = Callable[[str], Any]


def _default_session(path: str, threads: Optional[int] = None) -> Any:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        opts.intra_op_num_threads = int(threads)
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


class HayaiOCR:
    """Greedy Hayai OCR v2.5 recognition of one crop at a time (thread-safe: runs are serialised).

    ``session_factory(path)`` builds an onnxruntime-like session (``run(None, feeds)``);
    tests pass fakes."""

    def __init__(self, vision_path: str, decoder_path: str, tokenizer_path: str, *,
                 max_patches: int = DEFAULT_MAX_PATCHES, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                 threads: Optional[int] = None, session_factory: Optional[SessionFactory] = None) -> None:
        factory = session_factory or (lambda p: _default_session(p, threads))
        self.max_patches = int(max_patches)
        self.max_new_tokens = int(max_new_tokens)
        self.tokenizer = ByteLevelDecoder.from_file(tokenizer_path)
        self.vision = factory(vision_path)
        self.decoder = factory(decoder_path)
        self._lock = threading.Lock()

    def encode(self, image: Any) -> Tuple[np.ndarray, int, int]:
        """Vision features of a crop, pixel-unshuffled: ``([1, M, 3072], rows', cols')``."""
        rgb = _to_rgb_array(image)
        if rgb.size == 0 or rgb.shape[0] < 1 or rgb.shape[1] < 1:
            raise ValueError("empty image")
        pixels, rows, cols = patchify(rgb, self.max_patches)
        features = self.vision.run(None, {
            "p_grid": np.array([rows, cols], dtype=np.int64),
            "pixel_values": pixels[None],
        })[0]
        tokens, out_rows, out_cols = pixel_unshuffle(np.asarray(features, dtype=np.float32)[0], rows, cols)
        return tokens[None], out_rows, out_cols

    def generate(self, image: Any, *, stop: Optional[Callable[[], bool]] = None) -> List[int]:
        """Greedy token ids for a crop (without BOS / EOS)."""
        with self._lock:
            vision, rows, cols = self.encode(image)
            m = rows * cols
            cos, sin = rope_tables(rows, cols, 1, 0)
            logits, past = self.decoder.run(None, {
                "vision": vision,
                "tokens": np.array([[BOS_ID]], dtype=np.int64),
                "cos": cos,
                "sin": sin,
                "mask": prefill_mask(m),
                "past": np.zeros((N_KV, 1, KV_HEADS, 0, D_HEAD), dtype=np.float32),
            })
            out: List[int] = []
            empty = np.zeros((1, 0, vision.shape[-1]), dtype=np.float32)
            for step in range(1, self.max_new_tokens + 1):
                token = int(np.argmax(np.asarray(logits).reshape(-1)))
                if token in (EOS_ID, PAD_ID):
                    break
                out.append(token)
                if step == self.max_new_tokens or (stop is not None and stop()):
                    break
                cos, sin = rope_tables(0, 0, 1, step)
                length = int(past.shape[3])
                logits, past = self.decoder.run(None, {
                    "vision": empty,
                    "tokens": np.array([[token]], dtype=np.int64),
                    "cos": cos,
                    "sin": sin,
                    "mask": np.zeros((1, 1, 1, length + 1), dtype=np.float32),
                    "past": past,
                })
            return out

    def recognize(self, image: Any, *, stop: Optional[Callable[[], bool]] = None) -> str:
        """The text of a crop (PIL image or RGB array), stripped."""
        return self.tokenizer.decode(self.generate(image, stop=stop)).strip()


_CACHE_LOCK = threading.Lock()
_CACHE: Dict[Tuple[str, str, str, int, Optional[int]], HayaiOCR] = {}


def shared(vision_path: str, decoder_path: str, tokenizer_path: str, *, max_patches: int = DEFAULT_MAX_PATCHES,
           threads: Optional[int] = None) -> HayaiOCR:
    """One HayaiOCR per (files, patch budget, threads) per process (the sessions are ~600 MB)."""
    key = (os.path.abspath(vision_path), os.path.abspath(decoder_path), os.path.abspath(tokenizer_path),
           int(max_patches), threads)
    with _CACHE_LOCK:
        engine = _CACHE.get(key)
        if engine is None:
            engine = _CACHE[key] = HayaiOCR(vision_path, decoder_path, tokenizer_path,
                                            max_patches=max_patches, threads=threads)
        return engine


def release() -> int:
    """Drop the cached engines (``unload_models_after_translation``); returns how many."""
    with _CACHE_LOCK:
        count = len(_CACHE)
        _CACHE.clear()
    return count
