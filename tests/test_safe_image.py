import io

import pytest
from PIL import Image, UnidentifiedImageError

from safe_image import SAFE_IMAGE_FORMATS, open_image


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
