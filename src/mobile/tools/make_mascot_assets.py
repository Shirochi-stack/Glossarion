"""Build the app's mascot assets from the Halgakos source art in src/.

    python tools/make_mascot_assets.py            # writes app/assets/*.png
    python tools/make_mascot_assets.py --check    # exit 1 if the committed assets are stale

Outputs (all from the chibi src/Halgakos.png unless noted):
- icon.png      1024x1024 opaque on #121826 (launcher icon; matches [tool.flet.splash] colours)
- splash.png    1024x1024 opaque on #121826 (launch screen)
- halgakos_chibi.png   transparent, 512 px tall (empty states, Welcome, About, cover placeholder)
- halgakos_avatar.png  transparent square head crop, 256 px (CircleAvatar spots: chat, drawer)
- halgakos_full.png    transparent, 768 px tall, from src/Halgakos_NoChibi.png (Manga translator's
                       semi-transparent placeholder)

and, for the native extension's Android notifications (owner #22; the drawables replace the old page icon):
- res/drawable-{mdpi..xxxhdpi}/glossarion_native_ic_notification.png  the small (status-bar) icon: the
  chibi as a white-on-transparent silhouette, its dark line art cut out (Android tints it; only alpha counts)
- res/drawable-nodpi/glossarion_native_ic_large.png  the large icon: the whole colour chibi (as in
  src/Halgakos.ico) on a transparent square
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

from PIL import Image, ImageChops

MOBILE = Path(__file__).resolve().parents[1]
SRC = MOBILE.parent
ASSETS = MOBILE / "app" / "assets"
CHIBI = SRC / "Halgakos.png"
FULL = SRC / "Halgakos_NoChibi.png"
BACKGROUND = (0x12, 0x18, 0x26, 255)  # #121826, [tool.flet.splash] color / dark_color
ANDROID_RES = (MOBILE / "extensions" / "flet_glossarion_native" / "src" / "flutter" / "flet_glossarion_native"
               / "android" / "src" / "main" / "res")
NOTIFICATION_ICON = "glossarion_native_ic_notification.png"
NOTIFICATION_LARGE = "glossarion_native_ic_large.png"
#: the large icon: a transparent square (px) with the whole chibi at this fraction of its height
NOTIFICATION_LARGE_SIZE, NOTIFICATION_LARGE_HEIGHT = 256, 0.96
#: the small icon is 24 dp; px per density bucket
NOTIFICATION_SIZES = {"mdpi": 24, "hdpi": 36, "xhdpi": 48, "xxhdpi": 72, "xxxhdpi": 96}
#: of the small icon's square: the figure's height, and the luminance below which the line art is cut out
NOTIFICATION_HEIGHT = 0.92
LINE_ART_LUMINANCE = 60

# Fraction of the square canvas the figure's height takes. Android adaptive icons show the centre
# 66% safely; the chibi's head and body sit inside that box at 0.78 because the art is portrait.
ICON_HEIGHT = 0.78
SPLASH_HEIGHT = 0.62
# Head crop of the chibi for small round avatars: the top part of the trimmed figure.
AVATAR_TOP, AVATAR_HEIGHT = 0.0, 0.60


def _trimmed(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    box = image.getchannel("A").getbbox()
    return image.crop(box) if box else image


def _fit_height(image: Image.Image, height: int) -> Image.Image:
    scale = height / image.height
    return image.resize((max(1, round(image.width * scale)), height), Image.LANCZOS)


def _on_square(figure: Image.Image, size: int, height_fraction: float, background) -> Image.Image:
    fig = _fit_height(figure, round(size * height_fraction))
    if fig.width > size * 0.9:  # very wide art: fit the width instead
        scale = size * 0.9 / fig.width
        fig = fig.resize((round(fig.width * scale), round(fig.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), background)
    canvas.alpha_composite(fig, ((size - fig.width) // 2, (size - fig.height) // 2))
    return canvas


def build() -> dict:
    chibi = _trimmed(CHIBI)
    full = _trimmed(FULL)
    out = {
        "icon.png": _on_square(chibi, 1024, ICON_HEIGHT, BACKGROUND).convert("RGB"),
        "splash.png": _on_square(chibi, 1024, SPLASH_HEIGHT, BACKGROUND).convert("RGB"),
        "halgakos_chibi.png": _fit_height(chibi, 512),
        "halgakos_full.png": _fit_height(full, 768),
    }
    head = chibi.crop((0, round(chibi.height * AVATAR_TOP), chibi.width,
                       round(chibi.height * (AVATAR_TOP + AVATAR_HEIGHT))))
    side = max(head.width, head.height)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.alpha_composite(head, ((side - head.width) // 2, side - head.height))
    out["halgakos_avatar.png"] = square.resize((256, 256), Image.LANCZOS)
    return out


def _notification_icon(chibi: Image.Image, size: int) -> Image.Image:
    """White-on-transparent chibi, built at 4x and downscaled so the edges stay smooth."""
    big = size * 4
    fig = _fit_height(chibi, round(big * NOTIFICATION_HEIGHT))
    solid = fig.getchannel("A").point(lambda v: 255 if v >= 128 else 0)
    lines = fig.convert("L").point(lambda v: 255 if v < LINE_ART_LUMINANCE else 0)
    white = Image.new("RGBA", fig.size, (255, 255, 255, 255))
    white.putalpha(ImageChops.subtract(solid, lines))
    canvas = Image.new("RGBA", (big, big), (255, 255, 255, 0))
    canvas.alpha_composite(white, ((big - fig.width) // 2, (big - fig.height) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def build_android(assets: dict) -> dict:
    """The notification drawables (path relative to ANDROID_RES -> image)."""
    chibi = _trimmed(CHIBI)
    out = {f"drawable-{bucket}/{NOTIFICATION_ICON}": _notification_icon(chibi, size)
           for bucket, size in NOTIFICATION_SIZES.items()}
    # the whole chibi, like the desktop's Halgakos.ico (owner 2026-10-21: the head crop looked cut off)
    out[f"drawable-nodpi/{NOTIFICATION_LARGE}"] = _on_square(chibi, NOTIFICATION_LARGE_SIZE, NOTIFICATION_LARGE_HEIGHT,
                                                             (0, 0, 0, 0))
    return out


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if an asset differs from a fresh build")
    args = parser.parse_args(argv)
    stale = []
    assets = build()
    targets = [(ASSETS / name, image) for name, image in assets.items()]
    targets += [(ANDROID_RES / name, image) for name, image in build_android(assets).items()]
    for target, image in targets:
        name = target.name if target.parent == ASSETS else str(target.relative_to(ANDROID_RES))
        data = _png_bytes(image)
        if args.check:
            current = target.read_bytes() if target.is_file() else b""
            if hashlib.sha256(current).digest() != hashlib.sha256(data).digest():
                stale.append(name)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f"wrote {target.relative_to(MOBILE)} ({len(data) // 1024} KB)")
    if stale:
        print("stale mascot assets: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
