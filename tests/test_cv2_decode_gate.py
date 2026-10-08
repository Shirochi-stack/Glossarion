"""OpenCV decoder gate (owner-approved desktop-shared hardening, 2026-10-08).

OpenCV chooses its decoder from the file content, and its bundled OpenJPEG 2.5.3 (OSV-2025-219,
a heap write with no released fix), libtiff, OpenEXR and PNM / HDR / Sun raster parsers would
otherwise parse untrusted manga pages, EPUB / CBZ images and custom image-edit responses.
``safe_image.cv2_imread`` / ``cv2_imdecode`` hand OpenCV only PNG, JPEG, WEBP, BMP and GIF bytes
and return None (cv2's own failure value) for anything else without calling OpenCV.

* G (gate): refused signatures never reach cv2 (a spy module stands in for cv2); allowed ones are
  delegated with exactly the caller's arguments; unreadable paths keep cv2's own contract.
* P (pixels): with the real OpenCV, every allowed format decodes identically to raw
  ``cv2.imread`` / ``cv2.imdecode`` for every common flag, Unicode paths included; formats raw
  OpenCV would decode (TIFF, JPEG 2000, PNM) are refused by the gate.
* C (call sites): no shared ``src/*.py`` module calls ``cv2.imread`` / ``cv2.imdecode`` (or any
  other OpenCV image reader) directly; the manga modules import the gate.
* I (imports): safe_image imports without Qt, Pillow, OpenCV or numpy and parses as Python 3.10.
"""

import ast
import logging
import os
import struct
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

import safe_image
from safe_image import CV2_DECODE_FORMATS, cv2_imdecode, cv2_imread, sniff_image_format

SRC = Path(__file__).resolve().parents[1] / "src"


def _icns_with_jpeg2000_subimage():
    """An ICNS whose only icon (ic08, 256x256) is a JPEG 2000 stream."""
    jp2 = b"\x00\x00\x00\x0cjP  \r\n\x87\n" + bytes(64)
    entry = b"ic08" + struct.pack(">I", 8 + len(jp2)) + jp2
    return b"icns" + struct.pack(">I", 8 + len(entry)) + entry


#: Leading bytes of formats OpenCV can decode but must never see from untrusted input.
REFUSED = {
    "jp2": b"\x00\x00\x00\x0cjP  \r\n\x87\n" + bytes(32),
    "j2k": b"\xff\x4f\xff\x51\x00\x2f" + bytes(32),
    "tiff_le": b"II*\x00\x08\x00\x00\x00" + bytes(32),
    "tiff_be": b"MM\x00*\x00\x00\x00\x08" + bytes(32),
    "bigtiff": b"II+\x00\x08\x00\x00\x00" + bytes(32),
    "icns_jp2": _icns_with_jpeg2000_subimage(),
    "pbm": b"P1\n2 2\n0 1\n1 0\n",
    "pgm": b"P5\n2 2\n255\n" + bytes(4),
    "ppm": b"P6\n2 2\n255\n" + bytes(12),
    "pam": b"P7\nWIDTH 1\nHEIGHT 1\nDEPTH 1\nMAXVAL 255\nTUPLTYPE GRAYSCALE\nENDHDR\n\x00",
    "pfm": b"PF\n1 1\n-1.0\n" + bytes(12),
    "hdr": b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n" + bytes(4),
    "exr": b"\x76\x2f\x31\x01\x02\x00\x00\x00" + bytes(32),
    "sunraster": b"\x59\xa6\x6a\x95" + bytes(32),
    "avif": b"\x00\x00\x00\x1cftypavif" + bytes(32),
    "text": b"<html><body>not an image</body></html>",
    "empty": b"",
}

#: Minimal headers of the allowed formats (the gate only sniffs; a spy decodes).
ALLOWED = {
    "PNG": b"\x89PNG\r\n\x1a\n" + bytes(24),
    "JPEG": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + bytes(24),
    "WEBP": b"RIFF\x24\x00\x00\x00WEBPVP8 " + bytes(24),
    "BMP": b"BM" + bytes(40),
    "GIF": b"GIF89a" + bytes(24),
}


@pytest.fixture
def cv2_spy(monkeypatch):
    """A stand-in ``cv2`` module that records every decode call (the gate imports cv2 lazily)."""
    calls = []
    fake = types.ModuleType("cv2")
    fake.IMREAD_COLOR = 1
    fake.IMREAD_UNCHANGED = -1

    def imread(*args, **kwargs):
        calls.append(("imread", args, kwargs))
        return "decoded"

    def imdecode(*args, **kwargs):
        calls.append(("imdecode", args, kwargs))
        return "decoded"

    fake.imread, fake.imdecode = imread, imdecode
    monkeypatch.setitem(sys.modules, "cv2", fake)
    return calls


# ===========================================================================
# G: the gate
# ===========================================================================

@pytest.mark.parametrize("name", sorted(ALLOWED))
def test_sniff_recognises_the_allowed_signatures(name):
    assert sniff_image_format(ALLOWED[name]) == name
    assert sniff_image_format(memoryview(ALLOWED[name])) == name
    assert set(CV2_DECODE_FORMATS) == set(ALLOWED)


@pytest.mark.parametrize("head", [
    b"\x89PNG\r\n\x1a", b"\xff\xd8", b"RIFF\x24\x00\x00\x00WAVEfmt ", b"GIF88a", b"B", b"",
    b"\x00\x00\x00\x0cjP  \r\n\x87\n", b"II*\x00",
])
def test_sniff_rejects_near_misses_and_other_formats(head):
    assert sniff_image_format(head) is None


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_refused_bytes_never_reach_opencv(name, tmp_path, cv2_spy):
    np = pytest.importorskip("numpy")
    data = REFUSED[name]
    path = tmp_path / f"page_{name}.png"  # the extension does not matter: OpenCV sniffs content
    path.write_bytes(data)
    assert cv2_imread(str(path)) is None
    assert cv2_imread(path, -1) is None
    assert cv2_imdecode(data, 1) is None
    assert cv2_imdecode(np.frombuffer(data, dtype=np.uint8), 1) is None
    assert cv2_imdecode(bytearray(data), 1) is None
    assert cv2_spy == []


@pytest.mark.parametrize("name", sorted(ALLOWED))
def test_allowed_bytes_are_delegated_with_the_callers_arguments(name, tmp_path, cv2_spy):
    np = pytest.importorskip("numpy")
    path = str(tmp_path / f"page.{name.lower()}")
    with open(path, "wb") as handle:
        handle.write(ALLOWED[name])
    buf = np.frombuffer(ALLOWED[name], dtype=np.uint8)
    assert cv2_imread(path) == "decoded"
    assert cv2_imread(path, -1) == "decoded"
    assert cv2_imread(path, flags=1) == "decoded"
    assert cv2_imdecode(buf, 1) == "decoded"
    assert cv2_imdecode(buf, flags=-1) == "decoded"
    assert cv2_spy[:3] == [("imread", (path,), {}), ("imread", (path, -1), {}), ("imread", (path,), {"flags": 1})]
    assert cv2_spy[3][0] == "imdecode" and cv2_spy[3][1][0] is buf and cv2_spy[3][1][1:] == (1,)
    assert cv2_spy[4][0] == "imdecode" and cv2_spy[4][1] == (buf,) and cv2_spy[4][2] == {"flags": -1}


def test_non_contiguous_arrays_are_sniffed_too(cv2_spy):
    np = pytest.importorskip("numpy")
    tiff = np.frombuffer(REFUSED["tiff_le"] * 2, dtype=np.uint8)
    png = np.frombuffer(b"".join(bytes([b, 0]) for b in ALLOWED["PNG"]), dtype=np.uint8)
    assert cv2_imdecode(tiff[::1], 1) is None
    assert cv2_imdecode(png[::2], 1) == "decoded"  # a strided view whose bytes are a PNG header
    assert len(cv2_spy) == 1


def test_unreadable_path_keeps_cv2s_own_contract(tmp_path, cv2_spy):
    """Nothing to sniff, nothing for OpenCV to read either: cv2 gives its own warning and None."""
    missing = str(tmp_path / "missing.png")
    assert cv2_imread(missing) == "decoded"  # the spy answers; the real cv2 returns None
    assert cv2_imread(str(tmp_path)) == "decoded"  # a directory
    assert [c[1][0] for c in cv2_spy] == [missing, str(tmp_path)]


def test_embedded_nul_path_is_refused(cv2_spy):
    assert cv2_imread("page.png\x00.jp2") is None
    assert cv2_spy == []


def test_a_refusal_is_logged_once_per_source(tmp_path, cv2_spy, caplog):
    path = tmp_path / "scan.tif"
    path.write_bytes(REFUSED["tiff_le"])
    with caplog.at_level(logging.WARNING, logger="safe_image"):
        for _ in range(3):
            assert cv2_imread(str(path)) is None
    records = [r for r in caplog.records if r.name == "safe_image"]
    assert len(records) == 1
    assert "TIFF" in records[0].getMessage() and str(path) in records[0].getMessage()


# ===========================================================================
# P: pixels with the real OpenCV
# ===========================================================================

def _real_stack():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    return cv2, np, Image


def _pattern(np, height=19, width=23, channels=3, dtype="uint8"):
    yy, xx = np.mgrid[0:height, 0:width]
    top = 65535 if dtype == "uint16" else 255
    planes = [(xx * 37 + yy * 11 + k * 53) % (top + 1) for k in range(channels)]
    return np.stack(planes, axis=-1).astype(dtype).squeeze()


def _write_fixtures(directory, np, Image):
    """PNG (RGB, RGBA, L, 16-bit), JPEG (+ EXIF orientation), WEBP (lossy / lossless), BMP, GIF."""
    rgb = Image.fromarray(_pattern(np))
    out = {}

    def save(name, image, fmt, **kw):
        path = os.path.join(directory, name)
        image.save(path, format=fmt, **kw)
        out[name] = path

    save("rgb.png", rgb, "PNG")
    save("rgba.png", Image.fromarray(_pattern(np, channels=4)), "PNG")
    save("gray.png", rgb.convert("L"), "PNG")
    save("deep.png", Image.fromarray(_pattern(np, channels=1, dtype="uint16")), "PNG")
    save("plain.jpg", rgb, "JPEG", quality=90)
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 CW; OpenCV applies it unless IMREAD_IGNORE_ORIENTATION
    save("rotated.jpg", rgb, "JPEG", quality=90, exif=exif.tobytes())
    try:
        save("lossy.webp", rgb, "WEBP", quality=80)
        save("lossless.webp", Image.fromarray(_pattern(np, channels=4)), "WEBP", lossless=True)
    except (KeyError, OSError):  # a Pillow build without WebP
        pass
    save("rgb.bmp", rgb, "BMP")
    save("palette.bmp", rgb.convert("P"), "BMP")
    save("anim.gif", rgb.convert("P"), "GIF")
    return out


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and bool((a == b).all())


def _flags(cv2):
    names = ("IMREAD_COLOR", "IMREAD_UNCHANGED", "IMREAD_GRAYSCALE", "IMREAD_ANYDEPTH",
             "IMREAD_IGNORE_ORIENTATION", "IMREAD_REDUCED_COLOR_2")
    return [None] + [getattr(cv2, n) for n in names if hasattr(cv2, n)]


def test_allowed_formats_decode_pixel_identically(tmp_path):
    cv2, np, Image = _real_stack()
    fixtures = _write_fixtures(str(tmp_path), np, Image)
    decoded = 0
    for name, path in sorted(fixtures.items()):
        with open(path, "rb") as handle:
            head = handle.read(16)
        assert sniff_image_format(head) in CV2_DECODE_FORMATS, name
        buf = np.fromfile(path, dtype=np.uint8)
        for flag in _flags(cv2):
            args = () if flag is None else (flag,)
            raw = cv2.imread(path, *args)
            assert _same(cv2_imread(path, *args), raw), (name, flag)
            raw_buf = cv2.imdecode(buf, flag if flag is not None else cv2.IMREAD_COLOR)
            assert _same(cv2_imdecode(buf, flag if flag is not None else cv2.IMREAD_COLOR), raw_buf), (name, flag)
            decoded += raw is not None
    assert decoded > 0


def test_unicode_paths_behave_exactly_like_raw_opencv(tmp_path):
    """The gate reads the head with Python; cv2's own Unicode-path behaviour is left as it was
    (the callers' np.fromfile / PIL fallbacks still kick in where cv2 returns None)."""
    cv2, np, Image = _real_stack()
    directory = tmp_path / "漫画 ページ"
    directory.mkdir()
    path = str(directory / "頁 01.png")
    Image.fromarray(_pattern(np)).save(path, format="PNG")
    assert _same(cv2_imread(path), cv2.imread(path))


def _encode_with_pillow(Image, np, fmt, path):
    try:
        Image.fromarray(_pattern(np)).save(path, format=fmt)
    except (KeyError, OSError) as exc:  # no encoder in this Pillow build
        pytest.skip(f"Pillow cannot write {fmt}: {exc}")


@pytest.mark.parametrize("fmt,ext", [("TIFF", ".tif"), ("JPEG2000", ".jp2"), ("PPM", ".ppm")])
def test_formats_raw_opencv_decodes_are_refused(fmt, ext, tmp_path):
    cv2, np, Image = _real_stack()
    path = str(tmp_path / f"page{ext}")
    _encode_with_pillow(Image, np, fmt, path)
    if cv2.imread(path) is None:
        pytest.skip(f"this OpenCV build has no {fmt} decoder")
    assert cv2_imread(path) is None
    assert cv2_imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR) is None


# ===========================================================================
# C: every OpenCV image read in the shared modules goes through the gate
# ===========================================================================

#: OpenCV functions that decode encoded image bytes / files.
_CV2_READERS = ("imread", "imreadmulti", "imreadanimation", "imdecode", "imdecodemulti",
                "imdecodeanimation", "imcount")
#: Modules whose OpenCV page reads were routed through the gate (2026-10-08).
GATED_MODULES = {"ImageRenderer.py", "bubble_detector.py", "local_inpainter.py",
                 "manga_editor_core.py", "manga_translator.py"}


def _cv2_aliases(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "cv2":
                    names.add(alias.asname or "cv2")
    return names


def _shared_modules():
    return sorted(p for p in SRC.glob("*.py") if p.name != "safe_image.py")


def test_no_shared_module_calls_an_opencv_reader_directly():
    offenders = []
    for path in _shared_modules():
        source = path.read_text(encoding="utf-8", errors="replace")
        if "cv2" not in source:
            continue
        tree = ast.parse(source, filename=path.name)
        aliases = _cv2_aliases(tree) | {"cv2"}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in _CV2_READERS and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in aliases):
                offenders.append(f"{path.name}:{node.lineno} {node.func.value.id}.{node.func.attr}")
            if isinstance(node, ast.ImportFrom) and node.module == "cv2":
                for alias in node.names:
                    if alias.name in _CV2_READERS or alias.name == "*":
                        offenders.append(f"{path.name}:{node.lineno} from cv2 import {alias.name}")
    assert offenders == [], "route these through safe_image.cv2_imread / cv2_imdecode:\n" + "\n".join(offenders)


def test_gated_modules_import_the_gate_they_call():
    users = {}
    for path in _shared_modules():
        source = path.read_text(encoding="utf-8", errors="replace")
        if "cv2_imread" not in source and "cv2_imdecode" not in source:
            continue
        tree = ast.parse(source, filename=path.name)
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id in ("cv2_imread", "cv2_imdecode")}
        imported = {a.asname or a.name for n in tree.body if isinstance(n, ast.ImportFrom)
                    and n.module == "safe_image" for a in n.names}
        assert called <= imported, (path.name, called - imported)
        if called:
            users[path.name] = called
    assert set(users) == GATED_MODULES


@pytest.mark.parametrize("name", sorted(GATED_MODULES | {"safe_image.py"}))
def test_touched_modules_parse_as_python_310(name):
    ast.parse((SRC / name).read_text(encoding="utf-8"), filename=name, feature_version=(3, 10))


# ===========================================================================
# I: import hygiene
# ===========================================================================

def _run(code):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = os.pathsep.join([str(SRC)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True,
                          text=True, encoding="utf-8", timeout=120)


def test_safe_image_imports_without_qt_pillow_opencv_or_numpy():
    result = _run("""
        import sys
        for name in ('PySide6', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets', 'shiboken6'):
            sys.modules[name] = None
        import safe_image
        heavy = sorted(m for m in ('PIL', 'cv2', 'numpy') if m in sys.modules)
        assert heavy == [], heavy
        assert safe_image.sniff_image_format(b'\\x89PNG\\r\\n\\x1a\\n') == 'PNG'
        print('ok')
    """)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr


def test_safe_image_works_without_pillow():
    result = _run("""
        import sys
        sys.modules['PIL'] = None
        import safe_image
        assert safe_image.harden_pillow() is False
        assert safe_image.sniff_image_format(b'GIF89a') == 'GIF'
        print('ok')
    """)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr


# ===========================================================================
# W: Windows ANSI file names (OpenCV opens a non-ASCII path through the narrow C runtime)
# ===========================================================================

def _ansi_alias(name):
    """The file the C runtime's narrow fopen reaches for ``name`` (None: no such alias here)."""
    if os.name != "nt":
        return None
    try:
        alias = name.encode("utf-8").decode("mbcs")
    except UnicodeError:
        return None
    return None if alias == name else alias


needs_ansi_alias = pytest.mark.skipif(_ansi_alias("é.png") is None,
                                      reason="Windows with a single-byte ANSI code page only (CI: Linux)")


@needs_ansi_alias
@pytest.mark.parametrize("refused", ["tiff_le", "jp2", "ppm"])
def test_an_ansi_alias_of_a_unicode_page_never_reaches_opencv(refused, tmp_path, cv2_spy):
    """'é.png' is a PNG, but OpenCV's narrow fopen opens its ANSI alias ('Ã©.png'):
    a CBZ can ship both, so the alias must pass the gate too."""
    page = tmp_path / "é.png"
    page.write_bytes(ALLOWED["PNG"])
    alias = tmp_path / _ansi_alias(page.name)
    alias.write_bytes(REFUSED[refused])
    assert cv2_imread(str(page)) is None and cv2_spy == []
    reason = safe_image.cv2_refusal(str(page))
    assert reason and alias.name in reason and reason.endswith("not PNG, JPEG, WEBP, BMP or GIF")
    assert cv2_imread(str(alias)) is None and cv2_spy == []  # the alias itself is refused too


@needs_ansi_alias
def test_an_allowed_ansi_alias_and_a_lone_unicode_page_are_delegated(tmp_path, cv2_spy):
    page = tmp_path / "é.png"
    page.write_bytes(ALLOWED["PNG"])
    assert cv2_imread(str(page)) == "decoded"  # no alias file: OpenCV fails on its own (None)
    (tmp_path / _ansi_alias(page.name)).write_bytes(ALLOWED["JPEG"])
    assert cv2_imread(str(page)) == "decoded" and safe_image.cv2_refusal(str(page)) is None
    assert [c[1][0] for c in cv2_spy] == [str(page), str(page)]


@needs_ansi_alias
def test_the_real_opencv_never_decodes_an_ansi_alias(tmp_path):
    cv2, np, Image = _real_stack()
    page = tmp_path / "é.png"
    Image.fromarray(_pattern(np)).save(str(page), format="PNG")
    alias = tmp_path / _ansi_alias(page.name)
    _encode_with_pillow(Image, np, "TIFF", str(alias))
    if cv2.imread(str(page)) is None:
        pytest.skip("this OpenCV build cannot read the TIFF alias")
    assert cv2.imread(str(page)).shape[:2] == (19, 23)  # raw OpenCV decodes the alias (the bypass)
    assert cv2_imread(str(page)) is None


@pytest.mark.skipif(os.name != "nt", reason="the ANSI alias exists on Windows only")
def test_a_name_without_an_ansi_form_is_not_handed_to_opencv(monkeypatch, tmp_path, cv2_spy):
    page = tmp_path / "page.png"
    page.write_bytes(ALLOWED["PNG"])

    def no_ansi_form(path):
        raise UnicodeDecodeError("mbcs", b"\x81", 0, 1, "no mapping")

    monkeypatch.setattr(safe_image, "_windows_ansi_name", no_ansi_form)
    assert cv2_imread(str(page)) is None and cv2_spy == []
    assert safe_image.cv2_refusal(str(page)) is None  # a page Pillow can still read


def test_no_ansi_alias_off_windows_or_for_ascii_names(monkeypatch):
    assert safe_image._windows_ansi_name("page.png") is None
    monkeypatch.setattr(safe_image.os, "name", "posix")
    assert safe_image._windows_ansi_name("é.png") is None


# ===========================================================================
# R: why a page is refused (the manga pipeline logs it) and the Pillow fallback for pages
# ===========================================================================

def test_cv2_refusal_names_the_format(tmp_path):
    allowed = tmp_path / "ok.png"
    allowed.write_bytes(ALLOWED["PNG"])
    tiff = tmp_path / "page.png"
    tiff.write_bytes(REFUSED["tiff_le"])
    assert safe_image.cv2_refusal(str(allowed)) is None
    assert safe_image.cv2_refusal(str(tiff)) == "it is TIFF, not PNG, JPEG, WEBP, BMP or GIF"
    assert safe_image.cv2_refusal(str(tmp_path / "missing.png")) is None  # nothing to decode
    assert "NUL" in safe_image.cv2_refusal("page.png\x00.jp2")


def test_open_page_image_reads_only_the_gate_formats(tmp_path):
    _cv2, np, Image = _real_stack()
    from PIL import UnidentifiedImageError

    for fmt, ext in (("PNG", ".png"), ("JPEG", ".jpg"), ("BMP", ".bmp"), ("GIF", ".gif")):
        path = str(tmp_path / f"ok{ext}")
        Image.fromarray(_pattern(np)).save(path, format=fmt)
        with safe_image.open_page_image(path) as image:
            assert image.format == fmt and image.size == (23, 19)
    for fmt in ("TIFF", "PPM"):
        path = str(tmp_path / f"page_{fmt.lower()}.png")  # a page name the manga Files list accepts
        _encode_with_pillow(Image, np, fmt, path)
        with safe_image.open_image(path) as image:  # open_image keeps its own (wider) allowlist
            assert image.format == fmt
        with pytest.raises(UnidentifiedImageError):
            safe_image.open_page_image(path)


def test_each_refused_buffer_is_logged(cv2_spy, caplog, monkeypatch):
    monkeypatch.setattr(safe_image, "_refusals_logged", set())  # earlier tests refused the same bytes
    with caplog.at_level(logging.WARNING, logger="safe_image"):
        for name in ("tiff_le", "ppm", "icns_jp2", "tiff_le"):
            assert cv2_imdecode(REFUSED[name]) is None
    lines = [r.getMessage() for r in caplog.records if r.name == "safe_image"]
    assert len(lines) == 3 and "TIFF" in lines[0] and "PNM" in lines[1] and "ICNS" in lines[2]
    assert cv2_spy == []


def test_manga_process_image_refuses_a_page_before_any_work(tmp_path, monkeypatch):
    """MangaTranslator.process_image refuses a page that is not PNG / JPEG / WEBP / BMP / GIF up
    front, with the reason in the log: no OCR, no decoder, no "no text found" copy of the page."""
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    monkeypatch.delenv("GRACEFUL_STOP", raising=False)
    try:
        import manga_translator
    except Exception as exc:  # pragma: no cover - the manga stack is optional
        pytest.skip(f"manga_translator not importable: {exc}")
    page = tmp_path / "013_tiff.png"
    page.write_bytes(REFUSED["tiff_le"])
    output = tmp_path / "out" / "013_tiff.png"
    logged = []
    worked = []

    def fail(*args, **kwargs):
        worked.append(args)
        raise AssertionError("the refused page reached the pipeline")

    fake = types.SimpleNamespace(log_callback=None, batch_mode=False, batch_current=0, batch_size=0,
                                 _log=lambda message, level="info": logged.append((level, message)),
                                 _log_model_status=lambda: None, _block_if_over_cap=fail,
                                 detect_text_regions=fail, _check_stop=fail)
    result = manga_translator.MangaTranslator.process_image(fake, str(page), str(output))
    assert result["success"] is False and not output.exists() and worked == []
    assert result["errors"] == ["Not a supported image: 013_tiff.png (it is TIFF, not PNG, JPEG, WEBP, BMP or GIF)"]
    assert ("error", "❌ " + result["errors"][0]) in logged
