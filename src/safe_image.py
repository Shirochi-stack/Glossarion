"""Limit untrusted image parsing to formats supported by Glossarion.

Restrict plugins before Image.open parses headers: checking image.format after
opening is too late for malformed EPS headers. Keep Pillow's pixel-bomb guard.
Pillow must still be updated; this is not a replacement for dependency fixes.

Two more decoders sit behind this module (owner-approved hardening, 2026-10-08):

* Pillow's ICNS plugin decodes JPEG 2000 sub-images through OpenJPEG on its own.
  harden_pillow() switches that off; open_image() calls it before every open.
* OpenCV picks its decoder from the file content, so cv2.imread / cv2.imdecode
  would hand untrusted bytes to its bundled OpenJPEG (OSV-2025-219, unfixed),
  libtiff, OpenEXR or PNM / HDR / Sun raster parsers. cv2_imread() and
  cv2_imdecode() pass only PNG, JPEG, WEBP, BMP and GIF bytes to OpenCV
  (sniff_image_format) and return None, cv2's own failure value, for anything
  else without calling OpenCV. On Windows OpenCV opens a path through the
  narrow (ANSI) C runtime, so a non-ASCII name reaches a different file than
  Python's open(): the gate checks that ANSI-named file too.
* A manga page is one of those formats or nothing: cv2_refusal() says why a
  page is refused (the page pipeline logs it), and open_page_image() is the
  Pillow fallback for the same pages (Unicode paths), limited to the same
  formats, so a page OpenCV may not decode is not decoded by Pillow either.

Pillow, OpenCV and numpy are imported lazily, so importing this module stays
cheap and works without them (and without Qt). Python 3.10 compatible.
"""

import logging
import os
import sys

SAFE_IMAGE_FORMATS = (
    "AVIF", "BMP", "GIF", "ICNS", "ICO", "JPEG", "MPO", "PNG", "PPM",
    "TIFF", "WEBP", "XBM", "XPM",
)

#: The only formats OpenCV may decode from untrusted bytes (a magic-byte allowlist).
CV2_DECODE_FORMATS = ("PNG", "JPEG", "WEBP", "BMP", "GIF")

#: Leading bytes sniff_image_format looks at (the WEBP signature needs 12).
SNIFF_BYTES = 16

_logger = logging.getLogger(__name__)
_refusals_logged = set()
_REFUSAL_LOG_LIMIT = 100

#: Names for the log line only; the gate itself is the allowlist in sniff_image_format.
_REFUSED_NAMES = (
    (b"\x00\x00\x00\x0cjP  \r\n\x87\n", "JPEG 2000"),
    (b"\xff\x4f\xff\x51", "JPEG 2000 codestream"),
    (b"II*\x00", "TIFF"), (b"MM\x00*", "TIFF"), (b"II+\x00", "BigTIFF"), (b"MM\x00+", "BigTIFF"),
    (b"icns", "ICNS"),
    (b"\x76\x2f\x31\x01", "OpenEXR"),
    (b"#?", "Radiance HDR"),
    (b"\x59\xa6\x6a\x95", "Sun raster"),
)


def harden_pillow():
    """Stop Pillow's ICNS plugin from decoding JPEG 2000 sub-images (OpenJPEG).

    The plugin reads ``enable_jpeg2k`` when it decodes, so such an icon raises
    ValueError ("Unsupported icon subimage format") instead of reaching OpenJPEG;
    PNG and legacy ICNS sub-images keep working. Idempotent. Returns False when
    Pillow (or its ICNS plugin) is not installed.
    """
    try:
        from PIL import IcnsImagePlugin
    except ImportError:
        return False
    IcnsImagePlugin.enable_jpeg2k = False
    return True


def open_image(fp, mode="r", formats=None):
    """Open an image without enabling EPS, JPEG2000 or McIdas parsers.

    Import lazily so optional Pillow users retain their existing import behavior.
    An explicit format selection may narrow, but cannot widen, the allowed set.
    """
    from PIL import Image

    harden_pillow()
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


def sniff_image_format(head):
    """The CV2_DECODE_FORMATS name the leading bytes ``head`` belong to, or None.

    These are the signatures OpenCV itself checks, and no other OpenCV decoder
    claims bytes that start with one of them.
    """
    head = bytes(head[:SNIFF_BYTES])
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if head.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP"
    if head.startswith(b"BM"):
        return "BMP"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "GIF"
    return None


def _describe(head):
    for magic, name in _REFUSED_NAMES:
        if head.startswith(magic):
            return name
    if len(head) >= 2 and head[:1] == b"P" and head[1:2] in (b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"F", b"f"):
        return "PNM / PAM / PFM"
    return "an unsupported format (first bytes %s)" % (head[:8].hex() or "none")


def _log_refusal(source, head, key=None):
    """Warn once per source (bounded) that OpenCV was not given these bytes."""
    if key is None:
        try:
            key = os.fspath(source)
        except TypeError:
            key = repr(source)
    if key in _refusals_logged or len(_refusals_logged) >= _REFUSAL_LOG_LIMIT:
        return
    _refusals_logged.add(key)
    _logger.warning("Not decoding %s with OpenCV: it is %s, not PNG, JPEG, WEBP, BMP or GIF",
                    source, _describe(head))


def _buffer_size(buf):
    """The byte length of a bytes-like object or numpy array (-1 if unknown)."""
    try:
        return memoryview(buf).nbytes
    except (TypeError, ValueError):
        return int(getattr(buf, "nbytes", -1) or -1)


def _buffer_head(buf):
    """The first SNIFF_BYTES bytes of a bytes-like object or numpy array (b"" if unreadable)."""
    try:
        return memoryview(buf).cast("B")[:SNIFF_BYTES].tobytes()
    except (TypeError, ValueError):
        pass
    try:
        import numpy as np
        return np.asarray(buf).reshape(-1)[:SNIFF_BYTES].tobytes()
    except Exception:
        return b""


def _windows_ansi_name(path):
    """The file name OpenCV's narrow C-runtime ``fopen`` reaches for ``path`` on
    Windows, when it differs from the one Python's (wide) ``open()`` reads; else None.

    OpenCV receives a str path as UTF-8 bytes and the CRT reads those bytes in the
    ANSI code page, so ``"\u00e9.png"`` opens ``"\u00c3\u00a9.png"`` there (a bytes
    path goes to the CRT as it is). The ``mbcs`` codec is that conversion. Raises
    UnicodeError when the bytes have no ANSI reading (OpenCV cannot open such a name).
    """
    if os.name != "nt":
        return None
    name = os.fspath(path)
    if isinstance(name, bytes):
        narrow = name.decode("mbcs")
        return None if narrow == os.fsdecode(name) else narrow
    if name.isascii():
        return None
    narrow = name.encode("utf-8").decode("mbcs")
    return None if narrow == name else narrow


def _read_head(path):
    """The first SNIFF_BYTES of the file, None when Python cannot open it. Raises
    ValueError for an embedded NUL (OpenCV would read a truncated path)."""
    try:
        with open(path, "rb") as handle:
            return handle.read(SNIFF_BYTES)
    except OSError:
        return None


def _gated_names(path):
    """The file names to sniff for ``path``: its own, and on Windows the ANSI name
    OpenCV opens when that differs. Raises UnicodeError / ValueError / TypeError."""
    names = [path]
    alias = _windows_ansi_name(path)
    if alias is not None:
        names.append(alias)
    return names


def cv2_refusal(path):
    """Why the gate keeps the file at ``path`` from OpenCV, as a log phrase ("it is
    TIFF, not PNG, JPEG, WEBP, BMP or GIF"); None when it starts with a
    CV2_DECODE_FORMATS signature or cannot be opened (no decoder can read it then,
    and the caller reports that as before). On Windows the ANSI-named file OpenCV
    would open (``_windows_ansi_name``) must pass as well. Logs nothing.
    """
    try:
        names = _gated_names(path)
    except (UnicodeError, TypeError):
        return None  # no ANSI name: OpenCV cannot open it; Pillow reads the page itself
    except ValueError:
        return "the file name has an embedded NUL character"
    for index, name in enumerate(names):
        try:
            head = _read_head(name)
        except ValueError:
            return "the file name has an embedded NUL character"
        if head is not None and sniff_image_format(head) is None:
            if index:
                what = "OpenCV would open %s for it, which is %s" % (
                    os.path.basename(os.fsdecode(name)), _describe(head))
            else:
                what = "it is %s" % _describe(head)
            return what + ", not PNG, JPEG, WEBP, BMP or GIF"
    return None


def cv2_imread(path, *args, **kwargs):
    """``cv2.imread`` for files that may hold untrusted bytes.

    Returns None (cv2's own failure value) without calling OpenCV unless the file
    starts with a CV2_DECODE_FORMATS signature; otherwise returns exactly
    ``cv2.imread(path, *args, **kwargs)`` (so cv2's Unicode-path limits, flags and
    EXIF handling are unchanged). A path Python cannot open (missing, a directory,
    no permission) goes to cv2.imread as before: OpenCV cannot read it either and
    keeps its own warning and None. On Windows the ANSI-named file OpenCV opens for
    a non-ASCII path must start with an allowed signature too
    (``_windows_ansi_name``); a name without an ANSI form returns None (OpenCV
    cannot open it; callers fall back to Pillow, ``open_page_image``).
    """
    if isinstance(path, (str, bytes, os.PathLike)):
        try:
            names = _gated_names(path)
        except (UnicodeError, ValueError):
            return None
        for name in names:
            try:
                head = _read_head(name)
            except ValueError:  # an embedded NUL: OpenCV would read a truncated path
                return None
            if head is not None and sniff_image_format(head) is None:
                _log_refusal(name, head)
                return None
    import cv2
    return cv2.imread(path, *args, **kwargs)


def open_page_image(fp, mode="r"):
    """``open_image`` for a page the OpenCV gate also reads (the Pillow fallback
    for paths OpenCV cannot open, the editor's render): only CV2_DECODE_FORMATS,
    so a page the gate refuses is not decoded by Pillow either (it raises
    ``PIL.UnidentifiedImageError``, "cannot identify image file")."""
    return open_image(fp, mode=mode, formats=CV2_DECODE_FORMATS)


def cv2_imdecode(buf, *args, **kwargs):
    """``cv2.imdecode`` for untrusted bytes (bytes-like or a numpy array).

    Returns None without calling OpenCV unless ``buf`` starts with a
    CV2_DECODE_FORMATS signature; otherwise returns exactly
    ``cv2.imdecode(buf, *args, **kwargs)``.
    """
    head = _buffer_head(buf)
    if sniff_image_format(head) is None:
        size = _buffer_size(buf)
        # one line per distinct buffer (bounded), not one for the first refused buffer only
        _log_refusal("<image bytes, %d bytes>" % size, head, key=("<image bytes>", head, size))
        return None
    import cv2
    return cv2.imdecode(buf, *args, **kwargs)


# Pillow already loaded by the importer: harden it now. Otherwise open_image() does it
# before its first use (every Pillow open in Glossarion goes through open_image).
if "PIL" in sys.modules:
    harden_pillow()
