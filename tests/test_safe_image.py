import io
import os
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from safe_image import SAFE_IMAGE_FORMATS, harden_pillow, open_image

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF"])
def test_supported_images_decode(fmt):
    data = io.BytesIO()
    Image.new("RGB", (4, 3), "red").save(data, format=fmt)
    data.seek(0)
    with open_image(data) as image:
        image.load()
        assert image.size == (4, 3)


def test_restricted_plugin_is_not_invoked(monkeypatch):
    Image.init()
    called = []

    def forbidden_factory(*args, **kwargs):
        called.append(True)
        raise AssertionError("restricted parser ran")

    monkeypatch.setitem(Image.OPEN, "EPS", (forbidden_factory, lambda prefix: True))
    with pytest.raises(UnidentifiedImageError):
        open_image(io.BytesIO(b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 1 1\n"))
    assert called == []


def test_explicit_format_cannot_enable_restricted_parser(monkeypatch):
    seen = []
    monkeypatch.setattr(Image, "open", lambda fp, **kwargs: seen.append(kwargs["formats"]))
    open_image(io.BytesIO(), formats=["png", "EPS", "JPEG2000", "MCIDAS"])
    assert seen == [("PNG",)]


def test_pixel_bomb_guard_still_runs(monkeypatch):
    data = io.BytesIO()
    Image.new("RGB", (4, 3)).save(data, format="PNG")
    data.seek(0)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(Image.DecompressionBombError):
        open_image(data)


def test_restricted_formats_are_excluded():
    assert not {"EPS", "JPEG2000", "MCIDAS"}.intersection(SAFE_IMAGE_FORMATS)


def test_missing_optional_parser_does_not_break_png(monkeypatch):
    Image.init()
    monkeypatch.delitem(Image.OPEN, "AVIF", raising=False)
    data = io.BytesIO()
    Image.new("RGB", (4, 3)).save(data, format="PNG")
    data.seek(0)
    with open_image(data) as image:
        image.load()
        assert image.size == (4, 3)


def test_cover_validation_rejects_eps_before_its_parser(monkeypatch):
    from library_covers import _image_bytes_decodable, _probe_image_size

    Image.init()
    called = []

    def forbidden_factory(*args, **kwargs):
        called.append(True)
        raise AssertionError("cover validation entered the EPS parser")

    monkeypatch.setitem(Image.OPEN, "EPS", (forbidden_factory, lambda prefix: True))
    data = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 1 1\n"
    assert _image_bytes_decodable(data) is False
    assert _probe_image_size(data) is None
    assert called == []


# ---------------------------------------------------------------------------
# ICNS -> JPEG 2000 (owner-approved hardening, 2026-10-08). Pillow's ICNS plugin hands ic07-ic14 /
# icp4-icp6 sub-images that start with a JPEG 2000 signature to OpenJPEG whenever the codec is
# built in (desktop and iOS wheels), whatever the outer format allowlist says.
# ---------------------------------------------------------------------------

def _icns(subimage, kind=b"ic08"):
    entry = kind + struct.pack(">I", 8 + len(subimage)) + subimage
    return b"icns" + struct.pack(">I", 8 + len(entry)) + entry


JP2_SUBIMAGE = b"\x00\x00\x00\x0cjP  \r\n\x87\n" + bytes(64)


def _png_bytes(size):
    data = io.BytesIO()
    Image.new("RGBA", size, (10, 200, 30, 255)).save(data, format="PNG")
    return data.getvalue()


class _ReachedJpeg2000(Exception):
    pass


def _spy_jpeg2000(monkeypatch):
    from PIL import Jpeg2KImagePlugin

    built = []

    def spy(*args, **kwargs):
        built.append(True)
        raise _ReachedJpeg2000()

    monkeypatch.setattr(Jpeg2KImagePlugin, "Jpeg2KImageFile", spy)
    return built


def test_harden_pillow_turns_off_icns_jpeg2000():
    from PIL import IcnsImagePlugin

    assert harden_pillow() is True
    assert harden_pillow() is True  # idempotent
    assert IcnsImagePlugin.enable_jpeg2k is False


def test_icns_fixture_reaches_jpeg2000_without_the_hardening(monkeypatch):
    """Control: with Pillow's default flag the fixture does reach the JPEG 2000 plugin."""
    from PIL import IcnsImagePlugin, features

    if not features.check_codec("jpg_2000") or not hasattr(IcnsImagePlugin, "Jpeg2KImagePlugin"):
        pytest.skip("Pillow without OpenJPEG: the ICNS sub-image cannot reach it")
    built = _spy_jpeg2000(monkeypatch)
    IcnsImagePlugin.enable_jpeg2k = True
    try:
        with Image.open(io.BytesIO(_icns(JP2_SUBIMAGE)), formats=["ICNS"]) as image:
            with pytest.raises(_ReachedJpeg2000):
                image.load()
    finally:
        harden_pillow()
    assert built == [True]


def test_icns_jpeg2000_subimage_never_reaches_the_decoder(monkeypatch):
    from PIL import IcnsImagePlugin

    built = _spy_jpeg2000(monkeypatch)
    IcnsImagePlugin.enable_jpeg2k = True  # as a fresh Pillow with the codec starts; open_image resets it
    try:
        with open_image(io.BytesIO(_icns(JP2_SUBIMAGE))) as image:
            assert image.format == "ICNS"
            with pytest.raises(ValueError, match="Unsupported icon subimage format"):
                image.load()
    finally:
        harden_pillow()
    assert built == []
    assert IcnsImagePlugin.enable_jpeg2k is False


def test_icns_with_a_png_subimage_still_opens():
    with open_image(io.BytesIO(_icns(_png_bytes((256, 256))))) as image:
        image.load()
        assert image.format == "ICNS" and image.size == (256, 256)


def _run(code):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = os.pathsep.join([str(SRC)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True,
                          text=True, encoding="utf-8", timeout=120)


def test_import_hardens_a_pillow_the_importer_already_loaded():
    result = _run("""
        import PIL.Image
        import safe_image
        from PIL import IcnsImagePlugin
        assert IcnsImagePlugin.enable_jpeg2k is False
        print('ok')
    """)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr


def test_import_alone_leaves_pillow_unloaded_until_first_use():
    result = _run("""
        import io, sys
        import safe_image
        assert 'PIL' not in sys.modules
        image = io.BytesIO()
        from PIL import Image
        Image.new('RGB', (2, 2)).save(image, format='PNG')
        image.seek(0)
        safe_image.open_image(image).load()
        from PIL import IcnsImagePlugin
        assert IcnsImagePlugin.enable_jpeg2k is False
        print('ok')
    """)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr
