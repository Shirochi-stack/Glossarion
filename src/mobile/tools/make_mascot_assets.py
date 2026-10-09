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
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

from PIL import Image

MOBILE = Path(__file__).resolve().parents[1]
SRC = MOBILE.parent
ASSETS = MOBILE / "app" / "assets"
CHIBI = SRC / "Halgakos.png"
FULL = SRC / "Halgakos_NoChibi.png"
BACKGROUND = (0x12, 0x18, 0x26, 255)  # #121826, [tool.flet.splash] color / dark_color

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


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if an asset differs from a fresh build")
    args = parser.parse_args(argv)
    stale = []
    for name, image in build().items():
        data = _png_bytes(image)
        target = ASSETS / name
        if args.check:
            current = target.read_bytes() if target.is_file() else b""
            if hashlib.sha256(current).digest() != hashlib.sha256(data).digest():
                stale.append(name)
            continue
        target.write_bytes(data)
        print(f"wrote {target.relative_to(MOBILE)} ({len(data) // 1024} KB)")
    if stale:
        print("stale mascot assets: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
