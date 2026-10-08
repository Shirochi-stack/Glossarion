"""Acceptance test for the owner's device-fix issue 9 (devfix3, 2026-10-08): untrusted manga pages
cannot reach OpenJPEG / libtiff through OpenCV or Pillow's ICNS plugin (the owner-approved decoder
hardening in ``src/safe_image.py``), the pages are rejected with a clear message, and PNG / JPEG /
WEBP / BMP / GIF pages still translate exactly as before.

Crafted pages (real encodings where this host has the encoders, so raw OpenCV *would* hand them to
its bundled OpenJPEG / libtiff / PXM / Radiance parsers): a JPEG 2000 codestream, a JP2 file, an
LZW TIFF, an ICNS whose ``ic09`` icon is a JPEG 2000 stream, a binary PPM and a Radiance HDR file,
each saved under an image extension the Files tab accepts (``.png`` / ``.jpg`` / ``.webp``).
A spy wraps the real ``cv2`` readers (``imread`` / ``imdecode`` / ...: it records the leading bytes
each call was given), Pillow's decoder lookup (``Image._getdecoder``), every Pillow ``load()`` (format
and file) and ``Jpeg2KImageFile``.

* In this process: the shared ``safe_image`` paths (``cv2_imread`` / ``cv2_imdecode`` /
  ``open_image``, with and without the mobile contract), the desktop-shared manga modules that call
  them (``bubble_detector``, ``local_inpainter``, ``manga_editor_core`` bodies), and the mobile
  Tools › Manga screen (Files tab pick images / pick archive, Editor tab page loads) over the real
  ``MangaFileList`` and ``manga_editor_core.MangaEditorSession`` in the in-memory Flet session.
* In a subprocess (``runtime_bootstrap`` + ``diagnostics.e2e.E2ESession``: the real JobService,
  HeadlessOwner, ``manga_runner.HeadlessMangaRunner`` and the fake OpenAI server, exactly as the
  device E2E): Files add (folder + CBZ), Start (the ``manga`` job), the editor's Detect /
  Recognize / Clean / Translate steps (``manga_step`` jobs; the RT-DETR model is host_smoke's
  synthetic export), Import OCR, and Start reusing an imported OCR file.
* Regression: the device E2E ``e2e_manga_cbz`` (+ process hygiene) and ``host_smoke --checks
  manga_pipeline`` on a freshly collected bundle.
* Desktop (Qt): with PySide6 importable, ``ImageRenderer`` under offscreen Qt: the editor bodies
  re-bound into its namespace resolve the gate, and its own render worker refuses the pages.

Real data is never touched: every path the code may write to (OUTPUT_DIRECTORY, HOME, USERPROFILE,
APPDATA, GLOSSARION_LIBRARY_DIR, GLOSSARION_DATA_DIR, CONFIG_FILE, FLET_APP_STORAGE_*) points at
the test's tmp dir and HTTP logging is off.

Run from src/mobile with the mobile venv (``unset PYTHONPATH`` first)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue9.py

The module doubles as the subprocess driver (``python test_devfix_issue9.py --drive <json>`` /
``--desktop-qt <json>``); it imports only the standard library and pytest at module level.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import json
import logging
import os
import re
import struct
import subprocess
import sys
import threading
import traceback
import types
import zipfile
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
TOOLS_DIR = MOBILE_DIR / "tools"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_image_stack = pytest.mark.skipif(not (_has("cv2") and _has("numpy") and _has("PIL")),
                                       reason="OpenCV / numpy / Pillow not installed")
needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")
needs_cores = pytest.mark.skipif(not (_has("manga_files_core") and _has("manga_editor_core") and _has("manga_env")),
                                 reason="the shared manga cores are not importable")
E2E_DEPENDENCIES = ("ebooklib", "openai", "httpx", "lxml", "bs4", "tiktoken", "cv2", "numpy", "PIL", "onnxruntime")
SELFTEST_EPUB = APP_DIR / "assets" / "selftest" / "selftest_ko_12ch.epub"  # tools/prepare_assets.py (CI: prepare)
needs_e2e = pytest.mark.skipif(not SELFTEST_EPUB.is_file() or not all(_has(m) for m in E2E_DEPENDENCIES),
                               reason="the E2E needs tools/prepare_assets.py and the backend dependencies")

# ==========================================================================
# Pages
# ==========================================================================

#: The only content OpenCV may decode from an untrusted page (an independent copy of the
#: owner-approved allowlist, so a regression in safe_image's own sniffing cannot hide here).
ALLOWED_PREFIXES = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"BM", b"GIF87a", b"GIF89a")


def allowed_head(head: bytes) -> bool:
    head = bytes(head or b"")
    return head.startswith(ALLOWED_PREFIXES) or (head[:4] == b"RIFF" and head[8:12] == b"WEBP")


#: crafted page -> (file name with an extension the Files tab accepts, the format OpenCV would pick)
CRAFTED_FILES = {
    "j2k": ("011_j2k.jpg", "JPEG 2000 codestream (OpenJPEG)"),
    "jp2": ("012_jp2.png", "JP2 (OpenJPEG)"),
    "tiff": ("013_tiff.png", "TIFF, LZW (libtiff)"),
    "icns_jp2": ("014_icns.png", "ICNS with an ic09 JPEG 2000 icon (Pillow ICNS -> OpenJPEG)"),
    "pnm": ("015_pnm.jpg", "binary PPM (OpenCV PXM)"),
    "hdr": ("016_hdr.webp", "Radiance HDR (OpenCV RGBE)"),
}
#: allowed page format -> (file name, Pillow format)
ALLOWED_FILES = {
    "PNG": ("001_png.png", "PNG"),
    "JPEG": ("002_jpeg.jpg", "JPEG"),
    "WEBP": ("003_webp.webp", "WEBP"),
    "BMP": ("004_bmp.bmp", "BMP"),
    "GIF": ("005_gif.gif", "GIF"),
}
#: The fallback signatures when this host cannot encode a format (the gate still sees real magic).
_STATIC = {
    "j2k": b"\xff\x4f\xff\x51\x00\x2f\x00\x00" + bytes(64),
    "jp2": b"\x00\x00\x00\x0cjP  \r\n\x87\n\x00\x00\x00\x14ftypjp2 " + bytes(64),
    "tiff": b"II*\x00\x08\x00\x00\x00" + bytes(64),
    "pnm": b"P6\n4 4\n255\n" + bytes(48),
    "hdr": b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 2\n" + bytes(16),
}


def manga_page_png(index: int) -> bytes:
    from glossarion_mobile.diagnostics.fixtures import manga_page_png as build

    return build(index)


def _rgb(page_png: bytes):
    from safe_image import open_image

    with open_image(io.BytesIO(page_png)) as image:
        return image.convert("RGB")


def allowed_pages() -> dict:
    """{format: bytes}: five different manga pages (fixtures.manga_page_png) in the allowed formats."""
    out = {}
    for index, (fmt, (_name, pil_format)) in enumerate(ALLOWED_FILES.items(), start=1):
        png = manga_page_png(index)
        if pil_format == "PNG":
            out[fmt] = png
            continue
        buffer = io.BytesIO()
        image = _rgb(png)
        if pil_format == "GIF":
            image = image.convert("P")
        try:
            image.save(buffer, format=pil_format, **({"quality": 92} if pil_format in ("JPEG", "WEBP") else {}))
        except (KeyError, OSError):  # a Pillow build without this encoder
            continue
        out[fmt] = buffer.getvalue()
    return out


def crafted_pages() -> dict:
    """{name: bytes}: the crafted pages (real encodings of a manga page when this host can)."""
    image = _rgb(manga_page_png(7))
    out = dict(_STATIC)

    def encode(fmt, **kw):
        buffer = io.BytesIO()
        image.save(buffer, format=fmt, **kw)
        return buffer.getvalue()

    for name, fmt, kw in (("jp2", "JPEG2000", {}), ("j2k", "JPEG2000", {"no_jp2": True}),
                          ("tiff", "TIFF", {"compression": "tiff_lzw"}), ("pnm", "PPM", {})):
        try:
            out[name] = encode(fmt, **kw)
        except (KeyError, OSError, ValueError):
            pass
    try:
        import cv2
        import numpy as np

        ok, buf = cv2.imencode(".hdr", np.asarray(image, dtype=np.float32)[:, :, ::-1] / 255.0)
        if ok:
            out["hdr"] = buf.tobytes()
    except Exception:
        pass
    # ic09 = a 512x512 icon: Pillow's ICNS plugin hands it to Jpeg2KImagePlugin (OpenJPEG) unless
    # IcnsImagePlugin.enable_jpeg2k is off (safe_image.harden_pillow).
    try:
        icon = io.BytesIO()
        image.resize((512, 512)).save(icon, format="JPEG2000")
        jp2 = icon.getvalue()
    except (KeyError, OSError, ValueError):
        jp2 = _STATIC["jp2"]
    entry = b"ic09" + struct.pack(">I", 8 + len(jp2)) + jp2
    out["icns_jp2"] = b"icns" + struct.pack(">I", 8 + len(entry)) + entry
    return out


def write_pages(folder: Path, *, crafted: bool = True, allowed: bool = True) -> tuple:
    """Write the pages into ``folder``: ({format: path} allowed, {name: path} crafted)."""
    folder.mkdir(parents=True, exist_ok=True)
    good, bad = {}, {}
    if allowed:
        for fmt, data in allowed_pages().items():
            path = folder / ALLOWED_FILES[fmt][0]
            path.write_bytes(data)
            good[fmt] = str(path)
    if crafted:
        for name, data in crafted_pages().items():
            path = folder / CRAFTED_FILES[name][0]
            path.write_bytes(data)
            bad[name] = str(path)
    return good, bad


def build_crafted_cbz(path: Path) -> dict:
    """A CBZ of one good page and three crafted ones under .png names: {member: kind}."""
    crafted = crafted_pages()
    members = {"001.png": ("good", manga_page_png(9)), "002.png": ("jp2", crafted["jp2"]),
               "003.png": ("tiff", crafted["tiff"]), "004.png": ("icns_jp2", crafted["icns_jp2"])}
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for member, (_kind, data) in members.items():
            archive.writestr(member, data)
    return {member: kind for member, (kind, _data) in members.items()}


# ==========================================================================
# The decoder spy
# ==========================================================================

#: OpenCV functions that decode encoded image bytes or files.
CV2_READERS = ("imread", "imreadmulti", "imreadanimation", "imdecode", "imdecodemulti", "imdecodeanimation",
               "imcount")


def _norm(path) -> str:
    try:
        return os.path.normcase(os.path.abspath(os.fspath(path)))
    except TypeError:
        return ""


def _source_head(value) -> tuple:
    """(source, first 16 bytes hex or None when unreadable) of a cv2 reader's first argument."""
    if isinstance(value, (str, bytes, os.PathLike)) and not isinstance(value, bytes):
        try:
            with open(value, "rb") as handle:
                return _norm(value), handle.read(16).hex()
        except (OSError, ValueError):
            return _norm(value), None
    try:
        return "<buffer>", bytes(memoryview(value).cast("B")[:16]).hex()
    except (TypeError, ValueError):
        pass
    try:
        import numpy as np

        return "<buffer>", np.asarray(value).reshape(-1)[:16].tobytes().hex()
    except Exception:
        return "<buffer>", None


class DecoderSpy:
    """Records every OpenCV decode (with the leading bytes it was given), every Pillow decoder
    lookup and pixel load (format, file), and every JPEG 2000 image Pillow opens."""

    def __init__(self) -> None:
        self.events: list = []
        self._lock = threading.Lock()

    def _add(self, **event) -> dict:
        event["thread"] = threading.current_thread().name
        with self._lock:
            self.events.append(event)
        return event

    def mark(self) -> int:
        with self._lock:
            return len(self.events)

    def since(self, mark: int) -> list:
        with self._lock:
            return list(self.events[mark:])

    def install(self, patch) -> "DecoderSpy":
        """``patch(owner, name, value)``: pytest's monkeypatch.setattr, or setattr in a driver."""
        try:
            import cv2
        except ImportError:
            cv2 = None
        if cv2 is not None:
            for name in CV2_READERS:
                real = getattr(cv2, name, None)
                if callable(real):
                    patch(cv2, name, self._cv2(name, real))
        try:
            from PIL import Image, ImageFile, Jpeg2KImagePlugin
        except ImportError:
            return self
        real_getdecoder = Image._getdecoder

        def getdecoder(mode, decoder_name, args, extra=()):
            self._add(api="PIL._getdecoder", codec=str(decoder_name), mode=str(mode))
            return real_getdecoder(mode, decoder_name, args, extra)

        patch(Image, "_getdecoder", getdecoder)
        Image.init()
        classes = {ImageFile.ImageFile}
        for factory, _accept in list(Image.OPEN.values()):
            if isinstance(factory, type):
                classes.add(factory)
        for cls in classes:
            own = vars(cls).get("load")
            if callable(own):
                patch(cls, "load", self._load(cls, own))
        real_open = Jpeg2KImagePlugin.Jpeg2KImageFile._open

        def j2k_open(image):
            self._add(api="PIL.Jpeg2KImageFile", file=_norm(getattr(image, "filename", "") or ""))
            return real_open(image)

        patch(Jpeg2KImagePlugin.Jpeg2KImageFile, "_open", j2k_open)
        return self

    def _cv2(self, name, real):
        def reader(*args, **kwargs):
            source, head = _source_head(args[0] if args else kwargs.get("filename", kwargs.get("buf")))
            self._add(api=f"cv2.{name}", source=source, head=head)
            return real(*args, **kwargs)

        return reader

    def _load(self, cls, real):
        def load(image, *args, **kwargs):
            filename = getattr(image, "filename", "") or ""
            event = self._add(api="PIL.load", cls=cls.__name__, format=str(getattr(image, "format", "") or ""),
                              file=_norm(filename) if isinstance(filename, (str, os.PathLike)) and filename else "",
                              ok=None)
            try:
                result = real(image, *args, **kwargs)
            except BaseException as exc:
                event["ok"], event["error"] = False, f"{type(exc).__name__}: {exc}"[:200]
                raise
            event["ok"] = True
            return result

        return load


def opencv_forbidden(events: list) -> list:
    """OpenCV decodes of bytes outside the allowlist (an unreadable path decodes nothing)."""
    return [e for e in events if str(e.get("api", "")).startswith("cv2.") and e.get("head") is not None
            and not allowed_head(bytes.fromhex(e["head"]))]


def jpeg2000_decodes(events: list) -> list:
    return [e for e in events if e.get("api") == "PIL.Jpeg2KImageFile"
            or (e.get("api") == "PIL._getdecoder" and "jpeg2k" in str(e.get("codec", "")).lower())]


def pillow_loads_of(events: list, paths) -> list:
    """Pillow pixel loads of ``paths`` that decoded (a load the plugin refused does not count)."""
    wanted = {_norm(p) for p in paths}
    return [e for e in events if e.get("api") == "PIL.load" and e.get("file") in wanted and e.get("ok") is not False]


def cv2_reads_of(events: list, paths) -> list:
    wanted = {_norm(p) for p in paths}
    return [e for e in events if str(e.get("api", "")).startswith("cv2.") and e.get("source") in wanted]


# ==========================================================================
# Isolation
# ==========================================================================


@pytest.fixture
def iso(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in ("Output", "home", "Library", "data", "models", "appdata")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(paths["Output"]))
    monkeypatch.setenv("HOME", str(paths["home"]))
    monkeypatch.setenv("USERPROFILE", str(paths["home"]))
    monkeypatch.setenv("APPDATA", str(paths["appdata"]))
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(paths["Library"]))
    monkeypatch.setenv("GLOSSARION_DATA_DIR", str(paths["data"]))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.json"))
    for name, sub in (("BUBBLE_CACHE_DIR", "detector"), ("MODEL_CACHE_DIR", "inpainting"), ("ONNX_CACHE_DIR", "onnx")):
        monkeypatch.setenv(name, str(paths["models"] / sub))
    monkeypatch.delenv("GLOSSARION_MOBILE", raising=False)
    return paths


def _child_env(tmp_path: Path) -> dict:
    """A subprocess environment whose every writable location is under ``tmp_path``."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FLET_", "GLOSSARION_"))}
    for key in ("OUTPUT_DIRECTORY", "CONFIG_FILE", "BUBBLE_CACHE_DIR", "MODEL_CACHE_DIR", "ONNX_CACHE_DIR"):
        env.pop(key, None)
    for name in ("data", "cache", "temp", "home", "appdata", "localappdata", "Library", "Output", "work"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    env.update({f"FLET_APP_STORAGE_{name.upper()}": str(tmp_path / name) for name in ("data", "cache", "temp")})
    env.update(HOME=str(tmp_path / "home"), USERPROFILE=str(tmp_path / "home"), APPDATA=str(tmp_path / "appdata"),
               LOCALAPPDATA=str(tmp_path / "localappdata"), GLOSSARION_LIBRARY_DIR=str(tmp_path / "Library"),
               OUTPUT_DIRECTORY=str(tmp_path / "Output"), GLOSSARION_HTTP_LOG="0", PYTHONIOENCODING="utf-8",
               PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    return env


# ==========================================================================
# 1. The crafted pages are what the decoders would parse (fixture sanity)
# ==========================================================================


@needs_image_stack
def test_crafted_pages_carry_the_formats_raw_opencv_would_parse():
    import cv2
    import numpy as np

    pages = crafted_pages()
    assert set(pages) == set(CRAFTED_FILES)
    assert pages["j2k"].startswith(b"\xff\x4f\xff\x51")
    assert pages["jp2"].startswith(b"\x00\x00\x00\x0cjP  \r\n\x87\n")
    assert pages["tiff"][:4] in (b"II*\x00", b"MM\x00*")
    assert pages["icns_jp2"].startswith(b"icns") and pages["icns_jp2"][8:12] == b"ic09"
    assert pages["icns_jp2"][16:28] == b"\x00\x00\x00\x0cjP  \r\n\x87\n"
    assert pages["pnm"].startswith(b"P6") and pages["hdr"].startswith(b"#?")
    assert not any(allowed_head(data[:16]) for data in pages.values())
    # Raw OpenCV decodes them on this build (each skipped where the build has no such decoder):
    # without the gate, OpenJPEG / libtiff / PXM / RGBE would parse an attacker's page.
    decoded = {name: cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is not None
               for name, data in pages.items()}
    info = cv2.getBuildInformation()
    if "JPEG 2000:" in info and "JPEG 2000:                   NO" not in info and len(pages["jp2"]) > 200:
        assert decoded["jp2"] and decoded["j2k"], decoded
    assert not decoded["icns_jp2"]  # OpenCV has no ICNS reader: the ICNS risk is Pillow's
    assert set(allowed_pages()) >= {"PNG", "JPEG", "BMP", "GIF"}


# ==========================================================================
# 2. The shared safe_image paths (desktop and mobile contract)
# ==========================================================================


@needs_image_stack
@pytest.mark.parametrize("mobile", [False, True], ids=["desktop", "mobile"])
def test_safe_image_refuses_crafted_bytes_before_any_decoder(mobile, tmp_path, monkeypatch, iso):
    import cv2
    import numpy as np
    from PIL import IcnsImagePlugin, UnidentifiedImageError

    import safe_image

    if mobile:
        monkeypatch.setenv("GLOSSARION_MOBILE", "1")
        monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")
    good, bad = write_pages(tmp_path / "pages")
    spy = DecoderSpy().install(monkeypatch.setattr)
    for name, path in bad.items():
        data = Path(path).read_bytes()
        assert safe_image.cv2_imread(path) is None, name
        assert safe_image.cv2_imread(path, cv2.IMREAD_UNCHANGED) is None, name
        assert safe_image.cv2_imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None, name
        assert safe_image.cv2_imdecode(data, cv2.IMREAD_COLOR) is None, name
    assert [e for e in spy.events if e["api"].startswith("cv2.")] == []  # OpenCV never saw them
    # Pillow: the JPEG 2000 formats are not even identified; the ICNS icon refuses to reach OpenJPEG,
    # also after something switched the plugin back on (open_image re-applies the hardening).
    for name in ("j2k", "jp2", "hdr"):
        with pytest.raises(UnidentifiedImageError):
            safe_image.open_image(bad[name])
    monkeypatch.setattr(IcnsImagePlugin, "enable_jpeg2k", True)
    with safe_image.open_image(bad["icns_jp2"]) as icon:
        assert IcnsImagePlugin.enable_jpeg2k is False
        with pytest.raises(ValueError, match="Unsupported icon subimage format"):
            icon.load()
    assert jpeg2000_decodes(spy.events) == []
    # The allowed formats decode exactly as raw OpenCV does (path and buffer, every common flag).
    for fmt, path in good.items():
        buf = np.fromfile(path, dtype=np.uint8)
        for flag in (cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE):
            mark = spy.mark()
            gated = safe_image.cv2_imread(path, flag)
            raw = cv2.imread(path, flag)
            assert raw is not None and gated is not None and gated.dtype == raw.dtype, (fmt, flag)
            assert gated.shape == raw.shape and bool((gated == raw).all()), (fmt, flag)
            gated_buf = safe_image.cv2_imdecode(buf, flag)
            assert gated_buf.shape == raw.shape and bool((gated_buf == raw).all()), (fmt, flag)
            assert len([e for e in spy.since(mark) if e["api"].startswith("cv2.")]) == 3
    assert opencv_forbidden(spy.events) == []


@needs_image_stack
def test_icns_with_a_png_icon_still_opens(tmp_path):
    """The ICNS hardening only stops the JPEG 2000 icons: a PNG icon decodes as before."""
    from safe_image import open_image

    icon = io.BytesIO()
    _rgb(manga_page_png(1)).resize((512, 512)).save(icon, format="PNG")
    entry = b"ic09" + struct.pack(">I", 8 + len(icon.getvalue())) + icon.getvalue()
    path = tmp_path / "icon.icns"
    path.write_bytes(b"icns" + struct.pack(">I", 8 + len(entry)) + entry)
    with open_image(str(path)) as image:
        image.load()
        assert image.format == "ICNS" and image.size == (512, 512)


# ==========================================================================
# 3. Desktop-shared manga modules (no Qt): the moved bodies refuse the pages
# ==========================================================================


def _synthetic_rtdetr(cache: Path) -> str:
    """host_smoke's synthetic RT-DETR export where the download would land (nothing is fetched)."""
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    from host_smoke import SYNTHETIC_RTDETR_ONNX_B64

    cache.mkdir(parents=True, exist_ok=True)
    (cache / "detector.onnx").write_bytes(base64.b64decode(SYNTHETIC_RTDETR_ONNX_B64))
    (cache / "config.json").write_bytes(b"{}")
    return "detector.onnx"


def _reset_rtdetr(bd) -> None:
    cls = bd.BubbleDetector
    for name, value in (("_rtdetr_onnx_shared_session", None), ("_rtdetr_onnx_loaded", False),
                        ("_rtdetr_onnx_model_key", None), ("_rtdetr_onnx_model_path", None)):
        if hasattr(cls, name):
            setattr(cls, name, value)


@needs_image_stack
@pytest.mark.skipif(not (_has("bubble_detector") and _has("local_inpainter") and _has("onnxruntime")),
                    reason="bubble_detector / local_inpainter / onnxruntime not importable")
def test_shared_detector_and_inpainter_refuse_crafted_pages(tmp_path, monkeypatch, iso):
    import bubble_detector as bd
    import local_inpainter

    monkeypatch.setenv("GLOSSARION_MOBILE", "1")  # the Python onnxruntime RT-DETR path (phone)
    monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")
    good, bad = write_pages(tmp_path / "pages")
    detector = bd.BubbleDetector(config_path=str(tmp_path / "bubble_config.json"))
    filename = _synthetic_rtdetr(Path(detector.cache_dir))
    spy = DecoderSpy().install(monkeypatch.setattr)
    try:
        if not detector.load_rtdetr_onnx_model(onnx_filename=filename, force_reload=True):
            pytest.skip("the synthetic RT-DETR export does not load here")
        assert detector.detect_bubbles(good["PNG"], confidence=0.3, use_rtdetr=True)  # the path decodes PNG
        for name, path in bad.items():
            assert detector.detect_bubbles(path, confidence=0.3, use_rtdetr=True) == [], name
            assert detector.detect_with_rtdetr_onnx(image_path=path, confidence=0.3) in ([], {
                "bubbles": [], "text_bubbles": [], "text_free": []}), name
            assert detector.get_bubble_masks(path, [(1, 1, 5, 5)]) is None, name
            assert detector.visualize_detections(path, bubbles=[(1, 1, 5, 5)]) is None, name
            # the shared inpainter's own page read (its first statement; nothing else runs)
            assert local_inpainter.LocalInpainter.inpaint_with_bubble_detection(types.SimpleNamespace(), path) is None
    finally:
        _reset_rtdetr(bd)
    assert opencv_forbidden(spy.events) == []
    assert jpeg2000_decodes(spy.events) == []
    assert cv2_reads_of(spy.events, bad.values()) == []  # refused before OpenCV, not by it
    assert cv2_reads_of(spy.events, [good["PNG"]])


@needs_image_stack
@needs_cores
def test_moved_editor_bodies_refuse_crafted_pages(tmp_path, monkeypatch, iso):
    """The editor functions the desktop re-binds into ImageRenderer (the same code objects):
    OCR on regions, Clean, detection / inpainting sync helpers on a headless editor session."""
    import manga_editor_core as mec

    monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")
    for key in ("GRACEFUL_STOP", "TRANSLATION_CANCELLED", "WAIT_FOR_CHUNKS"):  # the bodies reset these flags
        monkeypatch.setenv(key, "0")
    monkeypatch.chdir(tmp_path)
    good, bad = write_pages(tmp_path / "pages")
    session = mec.MangaEditorSession(None, state_file=str(tmp_path / "state.json"),
                                     image_paths=list(bad.values()))
    lines: list = []
    session._log_callback = lambda text, level="info": lines.append(str(text))
    spy = DecoderSpy().install(monkeypatch.setattr)
    for name, path in bad.items():
        session.open_page(path)
        assert mec._run_ocr_on_regions(session, path, [{"bbox": [1, 1, 30, 30]}], {"provider": "custom-api"}) == []
        mec._run_clean_background(session, path, [])
        assert any(f"Failed to load image: {os.path.basename(path)}" in line for line in lines), (name, lines[-5:])
    session.close()
    assert opencv_forbidden(spy.events) == []
    assert jpeg2000_decodes(spy.events) == []
    assert cv2_reads_of(spy.events, bad.values()) == []


# ==========================================================================
# 4. Tools › Manga on the real screen: Files add + Editor page loads
# ==========================================================================


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


def _ui():
    return _load("_glossarion_devfix9_manga_ui", "test_manga_ui.py")


@needs_flet
@needs_cores
@needs_image_stack
def test_files_tab_add_and_editor_load_never_decode_crafted_pages(tmp_path, monkeypatch, iso):
    """Files › Add images / Add archive take the pages by extension (no decode; the run rejects
    them), and the Editor tab opens every page (header reads only: no OpenCV, no pixel decode,
    no JPEG 2000) over the real MangaFileList and MangaEditorSession."""
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    ui = _ui()
    monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")
    monkeypatch.chdir(tmp_path)
    good, bad = write_pages(tmp_path / "pages")
    cbz = tmp_path / "archive" / "crafted.cbz"
    members = build_crafted_cbz(cbz)
    picked = list(good.values()) + list(bad.values())
    spy = DecoderSpy().install(monkeypatch.setattr)
    store: dict = {}

    async def scenario():
        _conn, session = ui._tb()._fake_session("android")
        page = session.page
        files = ui.FakeFiles(picks=[picked, [str(cbz)]])
        ctx = ui._ctx(page, store, jobs=ui.FakeJobs(), files=files, output_root=str(iso["Output"]))
        manga = ui._session(tmp_path, store)
        screen = MangaScreen(parse_route("/tools/manga?tab=files"), ctx, session=manga)
        ui._mount(page, screen.get_body())
        screen.did_show()
        await ui._settle()
        tab = screen.files_tab
        added = await tab.pick_images()
        assert added == len(picked), (added, ctx.notes[-3:])
        assert await tab.pick_archive() == len(members)
        listed = manga.files.files
        assert {os.path.basename(p) for p in listed} >= {os.path.basename(p) for p in picked}
        add_events = list(spy.events)
        editor = screen.editor_tab
        opened = {}
        for index, path in enumerate(listed):
            snap = await editor.open_page(index)
            assert manga.editor is not None, manga.editor_error
            opened[path] = snap
            assert snap.get("image_path") == os.path.abspath(path)
        screen.dispose()
        manga.editor.close()
        return add_events, opened, listed

    add_events, opened, listed = asyncio.run(scenario())
    crafted_listed = [p for p in listed if os.path.basename(p) in {CRAFTED_FILES[n][0] for n in CRAFTED_FILES}
                      or (os.path.basename(p) in members and members[os.path.basename(p)] != "good")]
    assert len(crafted_listed) == len(bad) + 3
    # Files add decodes nothing at all (the list is built from names; thumbnails are Flutter's)
    assert [e for e in add_events if e["api"].startswith("cv2.") or e["api"] in ("PIL.load", "PIL._getdecoder")] == []
    # Editor page loads: no OpenCV read of any page, no pixel decode of a crafted page, no JPEG 2000
    assert [e for e in spy.events if e["api"].startswith("cv2.")] == []
    assert pillow_loads_of(spy.events, crafted_listed) == []
    assert jpeg2000_decodes(spy.events) == []


# ==========================================================================
# 5. The mobile jobs in a subprocess (bootstrap + E2E sandbox + real JobService)
# ==========================================================================

#: Words a refusal message must use (one of them, on a line naming the page).
REASON = re.compile(r"failed to load image|cannot identify image file|unsupported|not png, jpeg, webp, bmp or gif|"
                    r"not a supported image|refus", re.I)
#: A refusal must not surface as a raw crash of a later step.
CRASH = re.compile(r"'NoneType' object has no attribute|object is not subscriptable|Traceback \(most recent call last\)")


#: A known, unrelated failure of the shared run (U8; manga_files_core / manga_runner, unchanged by the
#: decoder hardening): "Create CBZ at end" looks for ``<stem>_translated`` folders next to the run's
#: FIRST file, so a run whose first file is a CBZ page logs "Error creating CBZ file" with its
#: traceback (also with no crafted page at all). ``test_create_cbz_at_end_with_a_cbz_page_first``
#: tracks it; the refusal checks look at the rest of the log.
_CBZ_AT_END_FAILURE = re.compile(r"^.*Error creating CBZ file: No translated images found.*$\n"
                                 r"(?:^(?:Traceback \(most recent call last\):|  .*|\w*Error: .*)$\n?)*", re.M)


def _without_cbz_at_end_failure(log: str) -> str:
    return _CBZ_AT_END_FAILURE.sub("", log)


def _page_lines(log: str, name: str) -> list:
    return [line for line in log.splitlines() if name in line]


def _clear_refusal(log: str, name: str) -> bool:
    return any(REASON.search(line) for line in _page_lines(log, name))


def _stem_outputs(outputs, names) -> list:
    stems = {os.path.splitext(n)[0] for n in names}
    return [p for p in outputs or () if os.path.splitext(os.path.basename(p))[0] in stems]


@pytest.fixture(scope="module")
def drive(tmp_path_factory):
    """One run of the subprocess driver (``--drive``) shared by the job tests below."""
    if not SELFTEST_EPUB.is_file() or not all(_has(m) for m in E2E_DEPENDENCIES):
        pytest.skip("the E2E needs tools/prepare_assets.py and the backend dependencies")
    tmp_path = tmp_path_factory.mktemp("devfix9_drive")
    out = tmp_path / "drive.json"
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--drive", str(out)], cwd=str(tmp_path),
                          env=_child_env(tmp_path), capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=1500)
    tail = (proc.stderr or "")[-6000:] + (proc.stdout or "")[-2000:]
    assert out.is_file(), f"the driver wrote no report (exit {proc.returncode}):\n{tail}"
    report = json.loads(out.read_text(encoding="utf-8"))
    assert not report.get("exception"), report["exception"] + "\n" + tail
    report["_tmp"] = str(tmp_path)
    return report


@needs_e2e
def test_files_add_takes_pages_and_archive_without_decoding(drive):
    files_add = drive["files_add"]
    names = [os.path.basename(p) for p in files_add["files"]]
    for name, _what in CRAFTED_FILES.values():
        assert name in names
    assert sorted(m for m in names if m in ("001.png", "002.png", "003.png", "004.png")) == [
        "001.png", "002.png", "003.png", "004.png"]
    assert [e for e in files_add["events"] if e["api"].startswith("cv2.") or e["api"] == "PIL.load"] == []


@needs_e2e
def test_run_never_hands_crafted_pages_to_opencv_or_openjpeg(drive):
    for section in ("run", "run_imported"):
        events = drive[section]["events"]
        assert opencv_forbidden(events) == [], section
        assert jpeg2000_decodes(events) == [], section
        assert cv2_reads_of(events, drive["pages"]["crafted"].values()) == [], section


@needs_e2e
def test_run_translates_every_allowed_format_page(drive):
    """PNG / JPEG / WEBP / BMP / GIF pages (and the CBZ's good page) still translate: an output
    the size of the source with the text rendered on it (pixels differ)."""
    run = drive["run"]
    assert run["state"] == "DONE", run.get("error")
    pages = drive["pages"]
    for fmt, info in pages["allowed_outputs"].items():
        assert info["output"], f"{fmt}: no translated page ({info})"
        assert info["same_size"] and info["changed"], (fmt, info)
    assert pages["cbz_good_output"]["output"] and pages["cbz_good_output"]["changed"]
    result = run["result"]
    assert result["manga_completed"] >= len(pages["allowed"]) + 1, result


@needs_e2e
def test_run_rejects_crafted_pages_with_a_clear_message(drive):
    """Start: every crafted page fails, gets no 'translated' output (nothing copied or packed into
    the CBZ) and the job log names the page with the reason; no raw crash text."""
    run = drive["run"]
    log = run["log"]
    names = [CRAFTED_FILES[n][0] for n in drive["pages"]["crafted"]] + ["002.png", "003.png", "004.png"]
    problems = []
    copied = _stem_outputs(run["result"].get("manga_outputs"), names)
    if copied:
        problems.append(f"'translated' outputs for refused pages (copies of the refused bytes): "
                        f"{[os.path.basename(os.path.dirname(p)) + '/' + os.path.basename(p) for p in copied]}")
    if drive["pages"].get("cbz_out_crafted_members"):
        problems.append(f"the output CBZ packs refused pages: {drive['pages']['cbz_out_crafted_members']}")
    unclear = [n for n in names if not _clear_refusal(log, n)]
    if unclear:
        problems.append("no clear refusal line for: " + ", ".join(
            f"{n} ({' | '.join(_page_lines(log, n)[-2:])[:200]})" for n in unclear))
    crashes = sorted({m.group(0) for m in CRASH.finditer(_without_cbz_at_end_failure(log))})
    if crashes:
        problems.append(f"raw crash text in the job log: {crashes}")
    failed = run["result"].get("manga_failed")
    if failed != len(names):
        problems.append(f"manga_failed={failed}, expected {len(names)}")
    assert not problems, "\n".join(problems)


@needs_e2e
@pytest.mark.xfail(strict=True, reason=(
    "pre-existing (U8, desktop-shared manga_files_core._create_cbz_from_isolated_folders / "
    "manga_runner.cbz_paths, not part of the decoder hardening): 'Create CBZ at end' (on by default on "
    "mobile) looks next to the run's first file, so a run that starts with a CBZ page logs "
    "'Error creating CBZ file: No translated images found' and its traceback"))
def test_create_cbz_at_end_with_a_cbz_page_first(drive):
    assert "Error creating CBZ file" not in drive["run"]["log"]


@needs_e2e
def test_editor_steps_reject_crafted_pages(drive):
    """Detect (synthetic RT-DETR), Recognize, Clean and Translate on each crafted page: no OpenCV /
    OpenJPEG decode, no output, and the step log says the page could not be loaded."""
    steps = drive["steps"]
    assert steps["detect:good"]["regions"] > 0, steps["detect:good"]  # the detector really runs
    problems = []
    for key, step in steps.items():
        if key.endswith(":good"):
            continue
        name = os.path.basename(step["image"])
        if step.get("exception"):
            problems.append(f"{key}: {step['exception'][-300:]}")
            continue
        if opencv_forbidden(step["events"]) or jpeg2000_decodes(step["events"]):
            problems.append(f"{key}: decoder events {opencv_forbidden(step['events']) + jpeg2000_decodes(step['events'])}")
        if step["outputs"]:
            problems.append(f"{key}: outputs {step['outputs']}")
        if not _clear_refusal(step["log"], name):
            problems.append(f"{key}: no clear refusal line ({' | '.join(_page_lines(step['log'], name)[-2:])[:200]})")
    assert not problems, "\n".join(problems)


@needs_e2e
def test_import_ocr_renders_allowed_pages_and_rejects_crafted_ones(drive):
    """Editor › Import OCR (render=True): the good page is rendered from the imported text; a
    crafted page is not decoded by OpenCV, OpenJPEG or any other decoder, gets no rendered output,
    and the log says why."""
    imp = drive["import_ocr"]
    assert imp["state"] == "DONE", imp.get("error")
    assert imp["rendered"].get("good"), imp["rendered"]
    events = imp["events"]
    assert opencv_forbidden(events) == [] and jpeg2000_decodes(events) == []
    problems = []
    crafted = drive["import_pages"]
    decoded = pillow_loads_of(events, crafted.values())
    if decoded:
        problems.append("crafted pages decoded by Pillow instead of being refused: " + ", ".join(
            sorted({f"{os.path.basename(e['file'])} ({e['format']}, {e['cls']})" for e in decoded})))
    libtiff = [e for e in events if e.get("api") == "PIL._getdecoder" and e.get("codec") == "libtiff"]
    if libtiff:
        problems.append(f"libtiff ran {len(libtiff)} time(s) (Pillow's TIFF decoder) on the imported pages")
    rendered = sorted(k for k, v in imp["rendered"].items() if v and k != "good")
    if rendered:
        problems.append(f"rendered crafted pages: {rendered}")
    unclear = [n for n, p in crafted.items() if not _clear_refusal(imp["log"], os.path.basename(p))]
    if unclear:
        problems.append(f"no clear refusal line for: {unclear}")
    assert not problems, "\n".join(problems)


@needs_e2e
def test_run_reusing_imported_ocr_rejects_crafted_pages(drive):
    """Files › Start after Import OCR (the run skips detection and goes straight to the page
    load): crafted pages must still be refused, not decoded through the Pillow fallback."""
    run = drive["run_imported"]
    events = run["events"]
    problems = []
    decoded = pillow_loads_of(events, drive["imported_run_pages"]["crafted"].values())
    if decoded:
        problems.append("crafted pages decoded by Pillow (the 'Unicode path' fallback after the gate refused "
                        "them): " + ", ".join(sorted({f"{os.path.basename(e['file'])} ({e['format']})" for e in decoded})))
    libtiff = [e for e in events if e.get("api") == "PIL._getdecoder" and e.get("codec") == "libtiff"]
    if libtiff:
        problems.append(f"libtiff ran {len(libtiff)} time(s) (Pillow's TIFF decoder)")
    names = [os.path.basename(p) for p in drive["imported_run_pages"]["crafted"].values()]
    translated = _stem_outputs(run["result"].get("manga_outputs"), names)
    if translated:
        problems.append(f"crafted pages 'translated': {[os.path.basename(p) for p in translated]}")
    for fmt, info in drive["imported_run_pages"]["allowed_outputs"].items():
        if not (info["output"] and info["changed"]):
            problems.append(f"{fmt}: the allowed page was not translated ({info})")
    assert not problems, "\n".join(problems)


# ==========================================================================
# 6. Regression: the device E2E manga scenario and host_smoke manga_pipeline
# ==========================================================================


@needs_e2e
def test_e2e_manga_cbz_still_passes(tmp_path):
    out = tmp_path / "e2e.json"
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from glossarion_mobile.diagnostics import e2e; "
            "raise SystemExit(e2e.main(sys.argv[2:]))")
    proc = subprocess.run([sys.executable, "-c", code, str(APP_DIR), "--json", str(out), "--only", "e2e_manga_cbz",
                           "--only", "e2e_process_hygiene"], cwd=str(tmp_path), env=_child_env(tmp_path),
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    assert out.is_file(), (proc.stderr or "")[-4000:]
    result = json.loads(out.read_text(encoding="utf-8"))
    checks = {c["name"]: c for c in result["checks"]}
    assert set(checks) == {"e2e_manga_cbz", "e2e_process_hygiene"}
    failed = {n: c.get("error") or c.get("reason") for n, c in checks.items() if c["status"] != "pass"}
    assert not failed and proc.returncode == 0, json.dumps(failed, ensure_ascii=False) + (proc.stderr or "")[-3000:]
    manga = checks["e2e_manga_cbz"]["detail"]
    assert manga["pages"] == manga["ocr_requests"] == 3 and manga["translation_requests"] >= 1
    assert manga["cbz"].endswith("_translated.cbz") and len(manga["cbz_members"]) == 3


@pytest.mark.skipif(not all(_has(m) for m in ("cv2", "numpy", "PIL", "onnxruntime", "openai")),
                    reason="host_smoke manga_pipeline needs cv2 / numpy / Pillow / onnxruntime / openai")
def test_host_smoke_manga_pipeline_still_passes(tmp_path):
    out = tmp_path / "smoke.json"
    proc = subprocess.run([sys.executable, str(TOOLS_DIR / "host_smoke.py"), "--collect", "--simulate", "android",
                           "--checks", "manga_pipeline", "--work-dir", str(tmp_path / "work"), "--json", str(out)],
                          cwd=str(MOBILE_DIR), env=_child_env(tmp_path), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=900)
    tail = (proc.stdout or "")[-3000:] + (proc.stderr or "")[-3000:]
    assert proc.returncode == 0 and out.is_file(), tail
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["ok"], report.get("failures")
    checks = report.get("checks") or {}
    found = checks.get("manga_pipeline") if isinstance(checks, dict) else next(
        (c for c in checks if c.get("name") == "manga_pipeline"), None)
    assert found and found.get("status") == "pass", found
    runner = (found.get("detail") or {}).get("runner") or {}
    assert runner.get("requests", [])[:1] == ["vision"] and runner.get("output") == "001.png", runner


# ==========================================================================
# 7. Desktop (Qt): ImageRenderer's namespace resolves the gate
# ==========================================================================


@needs_image_stack
@pytest.mark.skipif(not _has("PySide6"), reason="PySide6 not installed (the desktop interpreter runs this)")
def test_desktop_image_renderer_refuses_crafted_pages(tmp_path):
    out = tmp_path / "qt.json"
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", GLOSSARION_HTTP_LOG="0", PYTHONIOENCODING="utf-8",
               OUTPUT_DIRECTORY=str(tmp_path / "Output"), CONFIG_FILE=str(tmp_path / "config.json"),
               HOME=str(tmp_path / "home"), USERPROFILE=str(tmp_path / "home"), APPDATA=str(tmp_path / "appdata"),
               GLOSSARION_LIBRARY_DIR=str(tmp_path / "Library"), GLOSSARION_DATA_DIR=str(tmp_path / "data"))
    env.pop("GLOSSARION_MOBILE", None)
    env.pop("GLOSSARION_NO_PROCESSES", None)
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--desktop-qt", str(out)], cwd=str(tmp_path),
                          env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    assert out.is_file(), (proc.stderr or "")[-4000:]
    report = json.loads(out.read_text(encoding="utf-8"))
    assert not report.get("exception"), report["exception"]
    assert report["image_renderer_gate"] and len(report["gate_users"]) >= 6, report.get("gate_users")
    assert report["gate_bound"] == [], report["gate_bound"]
    assert all(v is False for v in report["render_worker"].values()), report["render_worker"]
    assert all(v == [] for v in report["ocr_regions"].values()), report["ocr_regions"]
    assert opencv_forbidden(report["events"]) == [] and jpeg2000_decodes(report["events"]) == []


# ==========================================================================
# Subprocess drivers
# ==========================================================================


def _job_report(session, spy: DecoderSpy, outcome, mark: int) -> dict:
    snap = session.service.snapshot(outcome.job_id)
    try:
        with open(session.service.job_log_path(outcome.job_id), encoding="utf-8", errors="replace") as handle:
            log = handle.read()
    except OSError as exc:
        log = f"<no job log: {exc}>"
    return {"state": outcome.state, "error": outcome.error, "result": dict(getattr(snap, "result", None) or {}),
            "outputs": list(outcome.outputs), "log": log, "events": spy.since(mark)}


def _output_info(source: str, outputs, out_dir: str = "") -> dict:
    """The translated page for ``source`` among ``outputs`` and how it compares to the source:
    ``<folder>/<stem>_translated/<name>`` for a loose page, ``out_dir/<name>`` when given (a CBZ's
    page: ``<cbz folder>/<cbz stem>_translated/<name>``)."""
    from safe_image import open_image

    stem = os.path.splitext(os.path.basename(source))[0]
    folder = os.path.normcase(os.path.dirname(os.path.abspath(source)))

    def beside(path: str) -> bool:
        where = os.path.normcase(os.path.dirname(os.path.abspath(path)))
        if out_dir:
            return where == os.path.normcase(os.path.abspath(out_dir))
        return where.startswith(folder + os.sep)

    match = [p for p in outputs or () if os.path.splitext(os.path.basename(p))[0] == stem and beside(p)]
    info = {"output": match[0] if match else None, "same_size": False, "changed": False}
    if match and os.path.isfile(match[0]):
        with open_image(match[0]) as rendered, open_image(source) as original:
            info["same_size"] = rendered.size == original.size
            info["changed"] = rendered.convert("RGB").tobytes() != original.convert("RGB").tobytes()
    return info


def _ocr_document(pages: dict, root: str) -> dict:
    import manga_ocr_io

    region = {"text": "기사님 안녕하세요", "translated_text": "Hello, sir knight", "bbox": [60, 120, 300, 140],
              "vertices": [[60, 120], [360, 120], [360, 260], [60, 260]], "confidence": 1.0}
    doc_pages = [manga_ocr_io.make_page(path, [dict(region)], index=i, source_root=root)
                 for i, path in enumerate(pages.values())]
    return manga_ocr_io.create_document(doc_pages, workflow="manga", source_root=root)


def _drive(out_path: str) -> int:
    """The mobile jobs on the crafted pages (run in a subprocess by the ``drive`` fixture)."""
    os.environ["GLOSSARION_E2E_ISOLATED"] = "1"
    from glossarion_mobile import runtime_bootstrap as rb

    paths = rb.bootstrap()
    from glossarion_mobile.diagnostics import e2e
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MANGA_OCR_TEXT

    report: dict = {}
    session = e2e.E2ESession(paths)
    spy = DecoderSpy()
    gate_log: list = []
    try:
        session.setup()
        import manga_ocr_io

        from glossarion_mobile.services import manga as svc

        session.configure("off", manga_ocr_provider="custom-api", manga_skip_inpainting=True,
                          manga_create_cbz_at_end=True, manga_glossary_enabled=False)
        settings = svc.merged_manga_settings(session.store.snapshot())
        settings.setdefault("ocr", {})["bubble_detection_enabled"] = False
        session.store.set("manga_settings", settings)
        session.store.flush()
        session._set_env("GLOSSARION_DATA_DIR", str(session._dir("data")))
        inbox = session.root / "Inbox" / "devfix9"
        good, bad = write_pages(inbox / "pages")
        cbz = inbox / "crafted.cbz"
        members = build_crafted_cbz(cbz)

        class _GateLog(logging.Handler):
            def emit(self, record):
                gate_log.append(record.getMessage())

        logging.getLogger("safe_image").addHandler(_GateLog())
        spy.install(setattr)
        report["pages"] = {"allowed": good, "crafted": bad, "cbz": str(cbz), "cbz_members": members}

        # 1. Files add (folder, then the CBZ)
        manga_root = session._dir("manga")
        files = svc.MangaFileList(session.store.snapshot(), save=session.store.set_many,
                                  temp_root=str(manga_root / "cbz"), config_source=session.store.snapshot,
                                  folders_root=str(manga_root / "folders"))
        mark = spy.mark()
        added = files.add_paths([str(inbox / "pages")]) + files.add_paths([str(cbz)])
        report["files_add"] = {"added": added, "files": list(files.files), "events": spy.since(mark)}

        # 2. Start (the manga job)
        mark = spy.mark()
        session.server.ocr_text = FAKE_MANGA_OCR_TEXT
        try:
            outcome = session.run_job(session.service, svc.batch_spec(files, output_root=str(session.root / "Output")),
                                      "devfix9 manga run")
        finally:
            session.server.ocr_text = None
        report["run"] = run = _job_report(session, spy, outcome, mark)
        outputs = run["result"].get("manga_outputs") or []
        report["pages"]["allowed_outputs"] = {fmt: _output_info(p, outputs) for fmt, p in good.items()}
        cbz_pages = [p for p in files.files if os.path.basename(p) in members]
        good_cbz = next(p for p in cbz_pages if members[os.path.basename(p)] == "good")
        report["pages"]["cbz_good_output"] = _output_info(good_cbz, outputs, str(cbz.parent / f"{cbz.stem}_translated"))
        with zipfile.ZipFile(cbz) as source_archive:
            crafted_bytes = {source_archive.read(m)[:64] for m, kind in members.items() if kind != "good"}
        crafted_bytes |= {Path(p).read_bytes()[:64] for p in bad.values()}
        packed = []
        for archive in run["result"].get("manga_cbz") or []:
            with zipfile.ZipFile(archive) as zf:
                for member in zf.namelist():
                    if not member.endswith("/") and zf.read(member)[:64] in crafted_bytes:
                        packed.append(f"{os.path.basename(archive)}:{member}")
        report["pages"]["cbz_out_crafted_members"] = packed

        # 3. Editor steps (manga_step jobs on one editor session); RT-DETR = host_smoke's synthetic export
        import bubble_detector as bd

        cache = Path(os.environ.get("BUBBLE_CACHE_DIR") or session._dir("detector"))
        filename = _synthetic_rtdetr(cache)
        real_load = bd.BubbleDetector.load_rtdetr_onnx_model

        def load_synthetic(self, model_id=None, force_reload=False, onnx_filename=None):
            return real_load(self, model_id=model_id, force_reload=force_reload, onnx_filename=filename)

        bd.BubbleDetector.load_rtdetr_onnx_model = load_synthetic
        steps_dir = inbox / "editor"
        editor_good, editor_bad = write_pages(steps_dir)
        es = svc.new_editor_session(image_paths=[editor_good["PNG"]] + list(editor_bad.values()),
                                    state_file=str(session._dir("data") / "image_state.json"))
        token = svc.register_editor_session(es)
        steps: dict = {}
        session.server.ocr_text = FAKE_MANGA_OCR_TEXT

        def run_step(key, step, image, **kw):
            mark = spy.mark()
            try:
                outcome = session.run_job(session.service, svc.step_spec(step, token, image, **kw), f"{key}", timeout=180)
                entry = _job_report(session, spy, outcome, mark)
                entry["outputs"] = list(entry["result"].get("manga_step_outputs") or [])
            except BaseException:
                entry = {"exception": traceback.format_exc(), "events": spy.since(mark), "outputs": [], "log": ""}
            entry["image"] = image
            steps[key] = entry
            return entry

        try:
            entry = run_step("detect:good", "detect", editor_good["PNG"])
            entry["regions"] = len(es.page_snapshot(editor_good["PNG"]).get("boxes") or [])
            for name, path in editor_bad.items():
                run_step(f"detect:{name}", "detect", path)
                es.open_page(path)
                if not es.boxes:
                    es.add_box(60, 120, 300, 140)
                for step in ("recognize", "clean", "translate"):
                    run_step(f"{step}:{name}", step, path)
        finally:
            session.server.ocr_text = None
        report["steps"] = steps

        # 4. Import OCR onto the editor pages (render=True)
        import_pages = {"good": editor_good["JPEG"]}
        import_pages.update(editor_bad)
        ocr_path = session._dir("data") / "devfix9_ocr.json"
        manga_ocr_io.write_document(str(ocr_path), _ocr_document(import_pages, str(steps_dir)))
        es.set_pages(list(import_pages.values()))
        mark = spy.mark()
        outcome = session.run_job(session.service, svc.step_spec("import_ocr", token, import_pages["good"],
                                                                  images=list(import_pages.values()),
                                                                  extra={"path": str(ocr_path)}),
                                  "devfix9 import OCR", timeout=300)
        imp = _job_report(session, spy, outcome, mark)
        imp["rendered"] = {name: (es.image_state_manager.get_state(os.path.abspath(path)) or {})
                           .get("rendered_image_path") for name, path in import_pages.items()}
        imp["rendered"] = {k: (v if v and os.path.isfile(v) else None) for k, v in imp["rendered"].items()}
        report["import_ocr"] = imp
        report["import_pages"] = editor_bad

        # 5. Start reusing an imported OCR file (fresh copies of the pages)
        again_good, again_bad = write_pages(inbox / "again")
        again = svc.MangaFileList(session.store.snapshot(), save=session.store.set_many,
                                  temp_root=str(manga_root / "cbz2"), config_source=session.store.snapshot,
                                  folders_root=str(manga_root / "folders2"))
        again.add_paths([str(inbox / "again")])
        doc_pages = dict(again_good)
        doc_pages.update(again_bad)
        again_ocr = session._dir("data") / "devfix9_again_ocr.json"
        manga_ocr_io.write_document(str(again_ocr), _ocr_document(doc_pages, str(inbox / "again")))
        mark = spy.mark()
        session.server.ocr_text = FAKE_MANGA_OCR_TEXT
        try:
            outcome = session.run_job(session.service, svc.batch_spec(again, output_root=str(session.root / "Output"),
                                                                      imported_ocr=str(again_ocr)),
                                      "devfix9 manga run (imported OCR)")
        finally:
            session.server.ocr_text = None
        report["run_imported"] = run2 = _job_report(session, spy, outcome, mark)
        report["imported_run_pages"] = {
            "allowed": again_good, "crafted": again_bad,
            "allowed_outputs": {fmt: _output_info(p, run2["result"].get("manga_outputs")) for fmt, p in again_good.items()}}
        report["gate_log"] = gate_log
        es.close()
    except BaseException:
        report["exception"] = traceback.format_exc()
        session.failed = True
    finally:
        session.close()
    Path(out_path).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


def _drive_desktop_qt(out_path: str) -> int:
    """ImageRenderer (desktop Qt module, offscreen) on the crafted pages."""
    report: dict = {}
    try:
        import tempfile

        import manga_editor_core as mec
        import safe_image

        import ImageRenderer as ir

        work = Path(tempfile.mkdtemp(prefix="devfix9_qt_", dir=os.getcwd()))
        os.chdir(work)
        good, bad = write_pages(work / "pages")
        spy = DecoderSpy().install(setattr)
        def names(code):
            found = set(code.co_names)
            for const in code.co_consts:
                if isinstance(const, types.CodeType):
                    found |= names(const)
            return found

        report["image_renderer_gate"] = ir.cv2_imread is safe_image.cv2_imread
        users = [name for name in mec.EDITOR_FUNCTIONS if "cv2_imread" in names(getattr(ir, name).__code__)]
        report["gate_users"] = users
        report["gate_bound"] = [name for name in users
                                if getattr(ir, name).__globals__.get("cv2_imread") is not safe_image.cv2_imread
                                or "imread" in names(getattr(ir, name).__code__) - {"cv2_imread"}]
        fake = types.SimpleNamespace(manga_integration=types.SimpleNamespace(translator=object()))
        report["render_worker"] = {
            name: ir._render_with_manga_translator_thread_safe(fake, path, [], str(work / f"out_{name}.png"), path)
            for name, path in bad.items()}
        session = mec.MangaEditorSession(None, state_file=str(work / "state.json"), image_paths=list(bad.values()))
        report["ocr_regions"] = {name: ir._run_ocr_on_regions(session, path, [{"bbox": [1, 1, 20, 20]}],
                                                               {"provider": "custom-api"})
                                 for name, path in bad.items()}
        session.close()
        report["events"] = spy.events
    except BaseException:
        report["exception"] = traceback.format_exc()
    Path(out_path).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--drive":
        raise SystemExit(_drive(sys.argv[2]))
    if len(sys.argv) == 3 and sys.argv[1] == "--desktop-qt":
        raise SystemExit(_drive_desktop_qt(sys.argv[2]))
    raise SystemExit("usage: test_devfix_issue9.py --drive <json> | --desktop-qt <json>")
