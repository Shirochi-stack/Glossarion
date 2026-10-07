"""Limit untrusted image parsing to formats supported by Glossarion.

Restrict plugins before Image.open parses headers: checking image.format after
opening is too late for malformed EPS headers. Keep Pillow's pixel-bomb guard.
Pillow must still be updated; this is not a replacement for dependency fixes.
"""

SAFE_IMAGE_FORMATS = (
    "AVIF", "BMP", "GIF", "ICNS", "ICO", "JPEG", "MPO", "PNG", "PPM",
    "TIFF", "WEBP", "XBM", "XPM",
)


def open_image(fp, mode="r", formats=None):
    """Open an image without enabling EPS, JPEG2000 or McIdas parsers.

    Import lazily so optional Pillow users retain their existing import behavior.
    An explicit format selection may narrow, but cannot widen, the allowed set.
    """
    from PIL import Image

    allowed = SAFE_IMAGE_FORMATS
    if formats is not None:
        if not isinstance(formats, (list, tuple)):
            raise TypeError("formats must be a list or tuple")
        allowed = tuple(fmt.upper() for fmt in formats if fmt.upper() in SAFE_IMAGE_FORMATS)
    # Some formats (MPO, for example) are adopted by another parser rather than
    # registered directly. Optional decoder builds may omit AVIF too. Passing
    # an unregistered name to Pillow's formats argument can raise KeyError.
    Image.init()
    allowed = tuple(fmt for fmt in allowed if fmt in Image.OPEN)
    return Image.open(fp, mode=mode, formats=allowed)
