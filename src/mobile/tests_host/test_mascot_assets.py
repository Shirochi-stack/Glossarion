"""Mascot art (owner, 2026-10-09): the app icon, launch screen and in-app mascot are the CHIBI Halgakos
(src/Halgakos.png); the Manga translator shows the non-chibi art (src/Halgakos_NoChibi.png) as a faded
placeholder. The assets are built by tools/make_mascot_assets.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL.Image")

MOBILE = Path(__file__).resolve().parents[1]
ASSETS = MOBILE / "app" / "assets"
SRC = MOBILE.parent
BACKGROUND = (0x12, 0x18, 0x26)

sys.path.insert(0, str(MOBILE / "app"))


def _figure_aspect(image, background=None) -> float:
    """Width / height of the drawn figure: the alpha bounding box, or the non-background box."""
    if background is None:
        box = image.convert("RGBA").getchannel("A").getbbox()
    else:
        rgb = image.convert("RGB")
        mask = PIL.new("L", rgb.size, 0)
        mask.putdata([0 if all(abs(a - b) <= 6 for a, b in zip(px, background)) else 255 for px in rgb.getdata()])
        box = mask.getbbox()
    assert box, "empty image"
    return (box[2] - box[0]) / (box[3] - box[1])


def test_icon_and_splash_are_the_chibi_on_the_splash_colour():
    chibi = _figure_aspect(PIL.open(SRC / "Halgakos.png"))
    full = _figure_aspect(PIL.open(SRC / "Halgakos_NoChibi.png"))
    assert abs(chibi - full) > 0.1, (chibi, full)  # the two arts are told apart by their shape
    for name in ("icon.png", "splash.png"):
        image = PIL.open(ASSETS / name)
        assert image.size == (1024, 1024), (name, image.size)
        assert image.convert("RGB").getpixel((2, 2)) == BACKGROUND, name  # [tool.flet.splash] colour
        aspect = _figure_aspect(image, BACKGROUND)
        assert abs(aspect - chibi) < abs(aspect - full), (name, aspect, chibi, full)


def test_in_app_mascot_assets_are_transparent_and_exist():
    from glossarion_mobile.ui.components import empty_state as es

    for name in (es.HALGAKOS_ASSET, es.HALGAKOS_AVATAR, es.HALGAKOS_FULL):
        image = PIL.open(ASSETS / name)
        assert image.mode == "RGBA" and image.getpixel((0, 0))[3] == 0, name
    assert es.HALGAKOS_ASSET != "icon.png"  # the launcher icon is opaque; in-app art is transparent
    full = _figure_aspect(PIL.open(ASSETS / es.HALGAKOS_FULL))
    reference = _figure_aspect(PIL.open(SRC / "Halgakos_NoChibi.png"))
    assert abs(full - reference) < 0.02, (full, reference)
    assert 0 < es.PLACEHOLDER_OPACITY < 0.5


def test_no_ui_code_uses_the_launcher_icon_as_art():
    ui = MOBILE / "app" / "glossarion_mobile" / "ui"
    offenders = [str(p.relative_to(MOBILE)) for p in ui.rglob("*.py")
                 if '"icon.png"' in p.read_text(encoding="utf-8", errors="replace")]
    assert not offenders, offenders
