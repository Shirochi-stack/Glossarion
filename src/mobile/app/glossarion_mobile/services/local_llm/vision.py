"""Images for the on-device engine's vision models (OpenAI ``image_url`` parts, ``data:`` URIs only).

* ``image_count(messages)``: how many ``image_url`` parts a request carries;
* ``check_parts(messages)``: ``None`` or why the parts cannot run here (an unknown part type, an
  ``http(s)`` URL: the endpoint never fetches anything, so a request cannot make the app download);
* ``prepare_messages(messages)``: every image decoded with the shared ``safe_image.open_image`` (only
  the formats Glossarion accepts; a JPEG is decoded at a reduced scale with ``draft`` and turned upright
  by its EXIF orientation), its longest side scaled down to ``MAX_SIDE`` px and re-encoded as
  JPEG (stb_image in libmtmd reads JPEG/PNG, not WebP; a phone photo becomes a few hundred KB and the
  projector's work stays bounded);
* ``IMAGE_TOKENS``: the context estimate per image (Qwen3.5 at 896 px: 56 x 56 patches merged 2 x 2,
  about 800 tokens) for the context check before the projector runs.

Pure Python + Pillow (imported lazily); no Flet.
"""

from __future__ import annotations

import base64
import binascii
import io
from typing import Any, Optional, Sequence

__all__ = ["IMAGE_TOKENS", "MAX_IMAGE_BYTES", "MAX_SIDE", "VisionInputError", "check_parts", "image_count",
           "prepare_image_url", "prepare_messages"]

MAX_SIDE = 896
IMAGE_TOKENS = 800
JPEG_QUALITY = 90
#: one decoded image at most (the request body limit is the server's MAX_BODY_BYTES)
MAX_IMAGE_BYTES = 24 * 1024 * 1024
TEXT_TYPES = ("text", "input_text")
IMAGE_TYPES = ("image_url", "input_image")


class VisionInputError(ValueError):
    """An image part that cannot be used (the message is the 400 answer)."""


def _parts(messages: Sequence[Any]):
    for message in messages or ():
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    yield part


def _url_of(part: dict) -> str:
    value = part.get("image_url")
    if isinstance(value, dict):
        value = value.get("url")
    return str(value or "")


def image_count(messages: Sequence[Any]) -> int:
    return sum(1 for part in _parts(messages) if part.get("type") in IMAGE_TYPES)


def check_parts(messages: Sequence[Any]) -> Optional[str]:
    for part in _parts(messages):
        kind = part.get("type")
        if kind in (None, *TEXT_TYPES):
            continue
        if kind not in IMAGE_TYPES:
            return f"This on-device model cannot read {kind} parts"
        if not _url_of(part).startswith("data:"):
            return "On-device models take images as data: URLs only"
    return None


def _decode_data_url(url: str) -> bytes:
    head, sep, data = url.partition(",")
    if not sep or not head.startswith("data:"):
        raise VisionInputError("The image is not a data: URL")
    if ";base64" not in head:
        raise VisionInputError("The image data: URL is not base64")
    if len(data) > MAX_IMAGE_BYTES * 4 // 3 + 16:
        raise VisionInputError("The image is too large")
    try:
        return base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise VisionInputError("The image data is not valid base64") from exc


def prepare_image_url(url: str, max_side: int = MAX_SIDE) -> str:
    """``url`` (a ``data:`` URL) as a JPEG ``data:`` URL whose longest side is at most ``max_side``."""
    raw = _decode_data_url(url)
    try:
        from PIL import Image, ImageOps

        from safe_image import open_image
    except ImportError as exc:  # pragma: no cover - Pillow ships with the app
        raise VisionInputError(f"Images cannot be read in this build ({exc})") from exc
    try:
        with open_image(io.BytesIO(raw)) as image:
            if image.format == "JPEG":  # a phone photo decodes at 1/2..1/8 scale, never below max_side
                image.draft("RGB", (max_side, max_side))
            image.load()
            if getattr(image, "is_animated", False):
                image.seek(0)
            upright = ImageOps.exif_transpose(image)  # a camera's EXIF rotation (lost in the re-encode)
            frame = upright.convert("RGBA") if "A" in upright.getbands() or upright.mode == "P" \
                else upright.convert("RGB")
    except Exception as exc:
        raise VisionInputError(f"The image could not be read ({type(exc).__name__})") from exc
    if frame.mode == "RGBA":  # transparent areas become white, as a page would show them
        background = Image.new("RGB", frame.size, (255, 255, 255))
        background.paste(frame, mask=frame.split()[-1])
        frame = background
    if max(frame.size) > max_side:
        frame.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    frame.save(out, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def prepare_messages(messages: Sequence[Any], max_side: int = MAX_SIDE) -> list:
    """A copy of ``messages`` with every image part as a scaled-down JPEG ``image_url`` part."""
    out: list = []
    for message in messages or ():
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            out.append(message)
            continue
        parts: list = []
        for part in message["content"]:
            if isinstance(part, dict) and part.get("type") in IMAGE_TYPES:
                parts.append({"type": "image_url", "image_url": {"url": prepare_image_url(_url_of(part), max_side)}})
            elif isinstance(part, dict) and part.get("type") == "input_text":
                parts.append({"type": "text", "text": str(part.get("text") or "")})
            else:
                parts.append(part)
        out.append(dict(message, content=parts))
    return out
