"""library_covers: GUI-free Library cover extraction (moved verbatim from epub_library).

Shared GUI-free core (Glossarion mobile rewrite, milestone U5). ``epub_library``
imports PySide6 at module scope; the cover helpers below moved here byte-for-byte and
``epub_library`` re-imports every name, so desktop callers get the same function
objects:

* ``_extract_cover`` (OPF meta / cover-image / filename / first <img> / first image /
  ebooklib fallback), ``_extract_pdf_cover`` (first embedded image of the first five
  pages; ``fitz`` is optional: without PyMuPDF the function returns None as before when
  the import failed), ``_find_cover_in_dir``, ``_find_folder_cover`` and
  ``_find_halgakos_icon``;
* ``CoverLoaderMixin.run`` is ``epub_library._CoverLoader.run`` (the per-card cover
  chain) byte-for-byte; the Qt thread inherits it and :func:`resolve_book_cover` runs it
  on a plain job object (the plan's mixin rule).

Adapted (documented in tests/parity/DISCREPANCIES.md, U5):

* ``_download_remote_cover_image`` validated the downloaded bytes with
  ``QImage.fromData(data).isNull()``. That was a pure computation, so it now uses
  ``_image_bytes_decodable`` (Pillow decode limited to the formats Qt sniffs, with Qt's
  truncation rules; else an image-magic sniff; SVG documents are accepted like Qt's SVG
  image plugin does).
* ``_cover_cache_dir`` honours ``set_cover_cache_dir()`` (process-wide; Glossarion
  Mobile points it at the app cache through ``library_core.install_library_env``).
  Desktop never calls it, so the cache stays ``%TEMP%/Glossarion_CoverCache``.

Log records keep the ``epub_library`` logger name so desktop log routing is unchanged.

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

from safe_image import open_image

import hashlib
import io
import logging
import os
import platform
import re
import struct
import sys
import tempfile
import traceback

from epub_package import find_epub_opf_member

# Same logger as before the move (records keep the "epub_library" name).
logger = logging.getLogger("epub_library")

#: Process-wide cover cache folder (None = ``%TEMP%/Glossarion_CoverCache``); see
#: :func:`set_cover_cache_dir`.
_COVER_CACHE_DIR_OVERRIDE = None


def set_cover_cache_dir(path) -> None:
    """Point the cover cache at *path* (None restores the temp-dir default).

    Glossarion Mobile calls this once (through ``library_core.install_library_env``)
    so covers survive in the app cache instead of a temp folder the OS may purge
    between launches. Desktop never calls it.
    """
    global _COVER_CACHE_DIR_OVERRIDE
    _COVER_CACHE_DIR_OVERRIDE = os.path.abspath(str(path)) if path else None


# ---------------------------------------------------------------------------
# Image header probes (replace QImage / QImageReader for pure computations)
# ---------------------------------------------------------------------------

_IMAGE_MAGIC = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"BM",
    b"\x00\x00\x01\x00",  # ICO
    b"II*\x00",
    b"MM\x00*",
)

#: Pillow formats that Qt's image readers recognise by content (``QImage.fromData`` /
#: ``QImageReader`` without a format hint). Formats without a signature (TGA, PCX, ...)
#: open in Pillow but not in Qt, so they count as unreadable here too.
_QT_SNIFFED_FORMATS = frozenset({
    "BMP", "GIF", "ICNS", "ICO", "JPEG", "MPO", "PNG", "PPM", "TIFF", "WEBP", "XBM", "XPM",
})
#: Of those, the formats Qt still returns a (partially decoded, non-null) image for
#: when the data is truncated (Pillow raises on ``load()``): the bytes of pixel data Qt
#: needs past the header (Pillow's tile offset) before the image is non-null.
_QT_PARTIAL_MIN_DATA = {"BMP": 1, "GIF": 2, "JPEG": 0, "MPO": 0, "XBM": 7}


def _looks_like_svg(data: bytes) -> bool:
    head = bytes(data[:4096]).lstrip()
    if head.startswith(b"\xef\xbb\xbf"):
        head = head[3:].lstrip()
    low = head.lower()
    return low.startswith(b"<svg") or (
        (low.startswith(b"<?xml") or low.startswith(b"<!doctype svg")
         or low.startswith(b"<!--")) and b"<svg" in low)


#: Qt SVG plugin unit factors for width / height (Qt treats pt and pc like px).
_SVG_UNIT_SCALE = {"": 1.0, "px": 1.0, "pt": 1.0, "pc": 1.0, "in": 90.0,
                   "mm": 3.543307, "cm": 35.43307}


def _svg_length(value, reference=None):
    """An SVG width / height the way Qt's SVG plugin reads it (None = unusable)."""
    match = re.match(
        r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(px|pt|pc|mm|cm|in|%)?\s*$",
        str(value or ""))
    if not match:
        return None
    number = float(match.group(1))
    unit = (match.group(2) or "").lower()
    if unit == "%":
        return None if reference is None else number * reference / 100.0
    return number * _SVG_UNIT_SCALE[unit]


def _qt_round(value: float) -> int:
    return int(value + 0.5) if value >= 0 else -int(-value + 0.5)


def _svg_size(data: bytes):
    """Default size of an SVG document, mirroring ``QSvgRenderer.defaultSize``.

    Both ``width`` and ``height`` (absolute units, or percentages of the viewBox) win
    and are truncated; otherwise the viewBox size (rounded). Documents with neither
    return None (Qt measures the drawing's bounding box instead; such SVGs are only
    classified as not full-page here).
    """
    try:
        from xml.etree import ElementTree as ET
        root = ET.fromstring(bytes(data))
    except Exception:
        return None
    if str(root.tag).rsplit("}", 1)[-1].lower() != "svg":
        return None
    view_box = [p for p in re.split(r"[\s,]+", str(root.get("viewBox") or "").strip()) if p]
    vb = None
    if len(view_box) == 4:
        try:
            vb = (float(view_box[2]), float(view_box[3]))
        except ValueError:
            vb = None
    if root.get("width") is not None and root.get("height") is not None:
        width = _svg_length(root.get("width"), vb[0] if vb else None)
        height = _svg_length(root.get("height"), vb[1] if vb else None)
        if width is not None and height is not None:
            return int(width), int(height)
    if vb is not None:
        return _qt_round(vb[0]), _qt_round(vb[1])
    return None


def _jpeg_size(data: bytes):
    pos = 2
    size = len(data)
    while pos + 4 <= size:
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7 or marker == 0xFF:
            pos += 1 if marker == 0xFF else 2
            continue
        if pos + 4 > size:
            return None
        seg_len = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if pos + 9 > size:
                return None
            height, width = struct.unpack(">HH", data[pos + 5:pos + 9])
            return width, height
        if seg_len < 2:
            return None
        pos += 2 + seg_len
    return None


def _webp_size(data: bytes):
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    chunk = data[12:16]
    if chunk == b"VP8 " and len(data) >= 30:
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X" and len(data) >= 30:
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height
    return None


def _probe_image_size(data: bytes):
    """Return ``(width, height)`` from an image header, or None when unreadable.

    Replaces ``QImageReader(buffer).size()`` (epub_library ``_reader_image_is_sizeable``):
    PNG / GIF / JPEG / BMP / WebP headers are parsed directly, SVG uses its width/height
    (or viewBox) like Qt's SVG plugin, anything else goes through Pillow when installed
    (only for the formats Qt decodes by content; TGA, PCX, ... count as unreadable).
    """
    data = bytes(data or b"")
    if not data:
        return None
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24 and data[12:16] == b"IHDR":
            return struct.unpack(">II", data[16:24])
        if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
            return struct.unpack("<HH", data[6:10])
        if data[:3] == b"\xff\xd8\xff":
            return _jpeg_size(data)
        if data[:2] == b"BM" and len(data) >= 26:
            header_size = struct.unpack("<I", data[14:18])[0]
            if header_size == 12:
                width, height = struct.unpack("<HH", data[18:22])
            else:
                width, height = struct.unpack("<ii", data[18:26])
            return abs(width), abs(height)
        if data[:4] == b"RIFF":
            size = _webp_size(data)
            if size:
                return size
        if _looks_like_svg(data):
            return _svg_size(data)
    except Exception:
        return None
    try:
        from PIL import Image
        with open_image(io.BytesIO(data)) as image:
            if str(image.format or "").upper() not in _QT_SNIFFED_FORMATS:
                return None
            return tuple(image.size)
    except Exception:
        return None


def _image_bytes_decodable(data: bytes) -> bool:
    """True when ``QImage.fromData(data)`` would not be null (Qt-free).

    SVG documents parse with an ``<svg>`` root (Qt's SVG plugin). Otherwise Pillow
    decodes the data, mirroring Qt's readers: only formats Qt recognises by content
    count; truncated JPEG / GIF / BMP / XBM still count once some pixel data follows the
    header (Qt returns the partial image); a PNG needs its ``IEND`` chunk (libpng in Qt
    rejects a stream cut before it). Without Pillow an image signature is enough.
    Known gaps: Qt decodes some ICO files cut inside the icon directory (Pillow cannot
    open them) and a JPEG cut inside its SOS header.
    """
    data = bytes(data or b"")
    if not data:
        return False
    if _looks_like_svg(data):
        try:
            from xml.etree import ElementTree as ET
            return str(ET.fromstring(data).tag).rsplit("}", 1)[-1].lower() == "svg"
        except Exception:
            return False
    try:
        from PIL import Image
    except Exception:
        Image = None
    if Image is not None:
        try:
            image = open_image(io.BytesIO(data))
        except Exception:
            return False
        with image:
            fmt = str(image.format or "").upper()
            if fmt not in _QT_SNIFFED_FORMATS:
                return False
            try:
                offset = int(image.tile[0][2]) if image.tile else 0
            except Exception:
                offset = 0
            try:
                image.load()
            except Exception:
                need = _QT_PARTIAL_MIN_DATA.get(fmt)
                return need is not None and len(data) >= offset + need
        if fmt == "PNG":
            return b"IEND" in data
        return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return any(data.startswith(magic) for magic in _IMAGE_MAGIC)


# ---------------------------------------------------------------------------
# Moved verbatim from epub_library (U5)
# ---------------------------------------------------------------------------


def _cover_cache_dir() -> str:
    d = _COVER_CACHE_DIR_OVERRIDE or os.path.join(
        tempfile.gettempdir(), "Glossarion_CoverCache")
    os.makedirs(d, exist_ok=True)
    return d


_PDF_COVER_SCAN_PAGE_LIMIT = 5


def _extract_pdf_cover(pdf_path: str) -> str | None:
    """Extract and cache the first embedded image in *pdf_path*.

    At most the first five pages are inspected in document order and only
    image objects are decoded; the page itself is never rendered. This does
    not invoke Glossarion's PDF text/chapter extraction pipeline or alter
    translation progress entries.
    """
    pdf_path = os.path.abspath(str(pdf_path or ""))
    if not pdf_path.lower().endswith(".pdf") or not os.path.isfile(pdf_path):
        return None
    try:
        stat = os.stat(pdf_path)
        identity = "\0".join((
            "pdf-cover-first-image-v2",
            os.path.normcase(pdf_path),
            str(stat.st_size),
            str(stat.st_mtime_ns),
        ))
        name_hash = hashlib.md5(identity.encode("utf-8")).hexdigest()[:16]
        cache_dir = _cover_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        cached = os.path.join(cache_dir, f"{name_hash}_pdf_image1.png")
        if os.path.isfile(cached) and os.path.getsize(cached) > 0:
            return cached

        import fitz

        pixmap = None
        with fitz.open(pdf_path) as document:
            if document.needs_pass or document.page_count < 1:
                return None
            seen_xrefs: set[int] = set()
            page_limit = min(
                int(document.page_count), _PDF_COVER_SCAN_PAGE_LIMIT)
            for page_number in range(page_limit):
                page = document.load_page(page_number)
                for image_info in page.get_images(full=True):
                    xref = int(image_info[0] or 0)
                    if xref <= 0 or xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    try:
                        candidate = fitz.Pixmap(document, xref)
                        if (candidate.width <= 0 or candidate.height <= 0
                                or candidate.colorspace is None):
                            continue
                        # PNG cannot directly encode CMYK/DeviceN pixmaps.
                        # Convert those while leaving RGB/gray pixels native.
                        if candidate.n - int(candidate.alpha) > 3:
                            candidate = fitz.Pixmap(fitz.csRGB, candidate)
                        pixmap = candidate
                        break
                    except Exception:
                        logger.debug(
                            "Skipping unreadable PDF image xref %s in %s",
                            xref, pdf_path,
                        )
                if pixmap is not None:
                    break
        if pixmap is None:
            return None

        fd, temporary = tempfile.mkstemp(
            prefix=f"{name_hash}_", suffix=".png", dir=cache_dir)
        os.close(fd)
        try:
            pixmap.save(temporary)
            if os.path.getsize(temporary) <= 0:
                return None
            os.replace(temporary, cached)
        finally:
            try:
                if os.path.exists(temporary):
                    os.remove(temporary)
            except OSError:
                pass
        return cached
    except Exception as exc:
        logger.debug("PDF cover rendering failed for %s: %s\n%s",
                     pdf_path, exc, traceback.format_exc())
        return None


def _download_remote_cover_image(url: str) -> bytes | None:
    """Download and validate an HTTP(S) image referenced by a cover page."""
    try:
        from urllib.parse import urlparse
        from urllib.request import Request, urlopen

        parsed = urlparse(str(url or "").strip())
        if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
            return None

        request = Request(
            parsed.geturl(),
            headers={
                "User-Agent": "Mozilla/5.0 (Glossarion EPUB Reader)",
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            },
        )
        max_bytes = 32 * 1024 * 1024
        with urlopen(request, timeout=20) as response:
            length_header = response.headers.get("Content-Length", "")
            try:
                if length_header and int(length_header) > max_bytes:
                    return None
            except (TypeError, ValueError):
                pass
            data = response.read(max_bytes + 1)
        if not data or len(data) > max_bytes:
            return None

        # Some novel sites serve extensionless image URLs as
        # application/octet-stream. Validate the bytes themselves instead of
        # requiring an image MIME type or filename extension.
        if not _image_bytes_decodable(data):
            return None
        return data
    except Exception:
        logger.debug(
            "Remote cover download failed for %s: %s",
            url,
            traceback.format_exc(),
        )
        return None


def _find_halgakos_icon() -> str | None:
    """Locate the Halgakos.ico fallback icon."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "Halgakos.ico"),
        os.path.join(os.path.dirname(__file__), "Halgakos.png"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "Halgakos.png"),
    ]
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        candidates.insert(0, os.path.join(exe_dir, "Halgakos.ico"))
        candidates.insert(1, os.path.join(exe_dir, "Halgakos.png"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _extract_cover(epub_path: str) -> str | None:
    cache_dir = _cover_cache_dir()
    name_hash = hashlib.md5(epub_path.encode("utf-8")).hexdigest()[:12]
    cached = os.path.join(cache_dir, f"{name_hash}.jpg")
    if os.path.isfile(cached):
        # Guard against stale 0-byte caches left behind by crashed writes:
        # returning one of those to QPixmap produces a null pixmap and hides
        # the cover silently. Re-extract when the cache is clearly empty.
        try:
            if os.path.getsize(cached) > 0:
                return cached
            os.remove(cached)
        except OSError:
            pass

    # Fast path: read EPUB as a zip and extract cover via OPF metadata
    # This avoids the heavy ebooklib.read_epub() which fully parses the DOM
    try:
        import zipfile
        import posixpath
        from xml.etree import ElementTree as ET

        with zipfile.ZipFile(epub_path, "r") as zf:
            names = zf.namelist()
            names_set = set(names)
            cover_data = None

            # --- Step 1: Find and parse the OPF file ---
            opf_path = find_epub_opf_member(zf)

            opf_dir = ""
            manifest_items = {}  # id -> (href, media_type)
            cover_meta_id = None

            if opf_path and opf_path in names_set:
                try:
                    opf_xml = zf.read(opf_path).decode("utf-8", errors="replace")
                    opf_tree = ET.fromstring(opf_xml)
                    opf_dir = posixpath.dirname(opf_path)

                    # Strip namespace for easier matching
                    opf_ns = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}

                    # Find cover image ID from <meta name="cover" content="..."/>
                    for meta_el in opf_tree.findall(".//{http://www.idpf.org/2007/opf}meta"):
                        if meta_el.get("name") == "cover":
                            cover_meta_id = meta_el.get("content")
                            break

                    # Build manifest lookup
                    for item_el in opf_tree.findall(".//{http://www.idpf.org/2007/opf}item"):
                        item_id = item_el.get("id", "")
                        item_href = item_el.get("href", "")
                        item_media = item_el.get("media-type", "")
                        item_props = item_el.get("properties", "")
                        full_href = posixpath.normpath(posixpath.join(opf_dir, item_href)) if item_href else ""
                        manifest_items[item_id] = (full_href, item_media, item_props)
                except Exception:
                    pass

            # --- Step 2: Try cover by OPF metadata ID ---
            if cover_meta_id and cover_meta_id in manifest_items:
                href, media_type, _ = manifest_items[cover_meta_id]
                if href in names_set and media_type.startswith("image/"):
                    cover_data = zf.read(href)

            # --- Step 3: Try cover by properties="cover-image" (EPUB3) ---
            if not cover_data:
                for item_id, (href, media_type, props) in manifest_items.items():
                    if "cover-image" in props and href in names_set:
                        cover_data = zf.read(href)
                        break

            # --- Step 4: Try images with "cover" in filename ---
            if not cover_data:
                img_exts = (".jpg", ".jpeg", ".png", ".gif", ".webp")
                for zname in names:
                    lower = zname.lower()
                    if any(lower.endswith(ext) for ext in img_exts) and "cover" in os.path.basename(lower):
                        cover_data = zf.read(zname)
                        break

            # --- Step 5: First <img> in the cover page / first HTML chapter ---
            if not cover_data:
                try:
                    from html import unescape
                    html_exts = (".xhtml", ".html", ".htm")
                    html_names = [
                        zname for zname in names
                        if any(zname.lower().endswith(ext) for ext in html_exts)
                    ]
                    # A cover document may only reference a remote image and
                    # therefore have no manifest image item. Prefer files such
                    # as cover.html before falling back to ordinary chapters.
                    html_names.sort(key=lambda zname: (
                        0 if "cover" in os.path.basename(zname).casefold() else 1,
                        zname.casefold(),
                    ))
                    for zname in html_names:
                        html = zf.read(zname).decode("utf-8", errors="replace")
                        img_match = re.search(
                            r"<img\b[^>]*\bsrc\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^'\"\s>]+))",
                            html,
                            re.IGNORECASE,
                        )
                        if img_match:
                            src = unescape(next(
                                (g for g in img_match.groups() if g),
                                "",
                            ))
                            if re.match(r"^https?://", src, re.IGNORECASE):
                                cover_data = _download_remote_cover_image(src)
                                if cover_data:
                                    break
                                continue
                            html_dir = posixpath.dirname(zname)
                            img_path = posixpath.normpath(
                                posixpath.join(html_dir, src)
                            )
                            if img_path in names_set:
                                cover_data = zf.read(img_path)
                                break
                except Exception:
                    pass

            # --- Step 6: First image file in the zip ---
            if not cover_data:
                img_exts = (".jpg", ".jpeg", ".png", ".gif", ".webp")
                for zname in names:
                    if any(zname.lower().endswith(ext) for ext in img_exts):
                        cover_data = zf.read(zname)
                        break

            if cover_data:
                with open(cached, "wb") as f:
                    f.write(cover_data)
                return cached
    except Exception as exc:
        logger.debug("Cover extraction (zipfile) failed for %s: %s\n%s", epub_path, exc, traceback.format_exc())

    # Last resort fallback: ebooklib (heavy, but handles edge cases)
    try:
        import ebooklib
        from ebooklib import epub as epub_mod

        book = epub_mod.read_epub(epub_path, options={"ignore_ncx": True})
        cover_data = None

        for meta in book.get_metadata("OPF", "cover"):
            if meta and meta[1]:
                cover_id = meta[1].get("content")
                if cover_id:
                    for item in book.get_items():
                        if item.get_id() == cover_id:
                            cover_data = item.get_content()
                            break
                break

        if not cover_data:
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_IMAGE and "cover" in item.get_name().lower():
                    cover_data = item.get_content()
                    break

        if not cover_data:
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_IMAGE:
                    cover_data = item.get_content()
                    break

        if cover_data:
            with open(cached, "wb") as f:
                f.write(cover_data)
            return cached
    except Exception as exc:
        logger.debug("Cover extraction (ebooklib) failed for %s: %s\n%s", epub_path, exc, traceback.format_exc())

    return None


def _find_cover_in_dir(folder: str) -> str | None:
    """Look for a cover image inside *folder* (or its ``images/`` subfolders).

    Preference order:
      1. *cover* images directly in the folder.
      2. Any image in the folder.
      3. *cover* images in ``images/`` or ``translated_images/``.
      4. The smallest-numbered image in ``images/``.
    """
    import re as _re
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

    def _natural_key(p):
        name = os.path.basename(p).lower()
        nums = _re.findall(r"\d+", name)
        return int(nums[0]) if nums else 0

    def _scan(dir_path: str) -> str | None:
        if not dir_path or not os.path.isdir(dir_path):
            return None
        covers: list[str] = []
        any_imgs: list[str] = []
        try:
            for entry in os.scandir(dir_path):
                if not entry.is_file(follow_symlinks=False):
                    continue
                nl = entry.name.lower()
                ext = os.path.splitext(nl)[1]
                if ext not in _IMG_EXTS:
                    continue
                if "cover" in nl:
                    covers.append(entry.path)
                any_imgs.append(entry.path)
        except (PermissionError, OSError):
            return None
        if covers:
            covers.sort(key=_natural_key)
            return covers[0]
        if any_imgs:
            any_imgs.sort(key=_natural_key)
            return any_imgs[0]
        return None

    direct = _scan(folder)
    if direct:
        return direct
    for sub in ("images", "translated_images"):
        r = _scan(os.path.join(folder, sub))
        if r:
            return r
    return None


def _find_folder_cover(file_path: str, config: dict | None = None, original_path: str | None = None) -> str | None:
    """Find a cover image for a PDF/TXT file.

    Search order:
      1. *cover* images in the file's own directory
      2. Original source path directory (from library_origins.txt)
      3. Output folder by base name — covers files moved to Library
    """
    import re as _re
    folder = os.path.dirname(file_path)

    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

    def _natural_key(p):
        name = os.path.basename(p)
        nums = _re.findall(r'\d+', name)
        return int(nums[0]) if nums else 0

    def _scan_for_cover(search_dir: str) -> str | None:
        """Look for *cover* images in search_dir, then any image in images/ subfolder."""
        if not os.path.isdir(search_dir):
            return None
        candidates = []
        try:
            for entry in os.scandir(search_dir):
                if entry.is_file(follow_symlinks=False):
                    nl = entry.name.lower()
                    ext = os.path.splitext(nl)[1]
                    if ext in _IMG_EXTS and "cover" in nl:
                        candidates.append(entry.path)
        except (PermissionError, OSError):
            pass
        if candidates:
            candidates.sort(key=_natural_key)
            return candidates[0]

        # Check images/ subfolder
        img_dir = os.path.join(search_dir, "images")
        if os.path.isdir(img_dir):
            img_cands = []
            try:
                for entry in os.scandir(img_dir):
                    if entry.is_file(follow_symlinks=False):
                        ext = os.path.splitext(entry.name.lower())[1]
                        if ext in _IMG_EXTS:
                            img_cands.append(entry.path)
            except (PermissionError, OSError):
                pass
            if img_cands:
                img_cands.sort(key=_natural_key)
                return img_cands[0]
        return None

    # 1. Check the file's own directory
    result = _scan_for_cover(folder)
    if result:
        return result

    # 2. Check original source path directory (persisted when moved to Library)
    if original_path:
        orig_dir = os.path.dirname(original_path)
        result = _scan_for_cover(orig_dir)
        if result:
            return result

    # 3. Check the original output folder by base name
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    config = config or {}
    output_dirs_to_check = []

    override = os.environ.get("OUTPUT_DIRECTORY") or config.get("output_directory")
    if override and os.path.isdir(override):
        output_dirs_to_check.append(os.path.join(os.path.abspath(override), base_name))

    # App directory (same logic as scan_for_epubs)
    if platform.system() == "Windows":
        if getattr(sys, "frozen", False):
            app_dir = os.path.dirname(sys.executable)
        else:
            app_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        app_dir = os.getcwd()
    output_dirs_to_check.append(os.path.join(app_dir, base_name))

    for out_dir in output_dirs_to_check:
        result = _scan_for_cover(out_dir)
        if result:
            return result

    return None


class CoverLoaderMixin:
    """``_CoverLoader``: a card's cover chain. Needs ``_file_path``, ``_file_type``,
    ``_config``, ``_original_path``, ``_raw_source_path``, ``_should_stop()`` and
    ``self.result_ready.emit(path, cover)``.
    """

    def run(self):
        if self._should_stop():
            return
        if self._file_type == "epub":
            cover = _extract_cover(self._file_path)
            if self._should_stop():
                return
            # Fallback 1: try the raw source EPUB (e.g. compiled EPUB in
            # Library/Translated may lack an embedded cover, but the
            # original raw EPUB typically has one).
            if not cover and self._raw_source_path:
                if (self._raw_source_path.lower().endswith(".epub")
                        and os.path.isfile(self._raw_source_path)):
                    cover = _extract_cover(self._raw_source_path)
                    if self._should_stop():
                        return
                elif (self._raw_source_path.lower().endswith(".pdf")
                      and os.path.isfile(self._raw_source_path)):
                    cover = _extract_pdf_cover(self._raw_source_path)
                    if self._should_stop():
                        return
            # Fallback 2: cover image sitting alongside the EPUB
            # (e.g. output folder with cover.jpg or images/ subfolder).
            if not cover:
                parent_dir = os.path.dirname(self._file_path)
                if parent_dir and os.path.isdir(parent_dir):
                    cover = _find_cover_in_dir(parent_dir)
                    if self._should_stop():
                        return
            # Fallback 3: broader search via original_path / output roots.
            if not cover:
                cover = _find_folder_cover(
                    self._file_path, config=self._config,
                    original_path=self._original_path)
                if self._should_stop():
                    return
        elif self._file_type == "in_progress":
            # For an in-progress card the "path" is the output folder itself.
            cover = None
            # Primary source: the resolved raw EPUB/PDF in Library/Raw (or
            # wherever source_epub.txt points). This is the only way to
            # produce a real thumbnail for Not Started cards, whose output
            # folder is still empty.
            if self._raw_source_path and os.path.isfile(self._raw_source_path):
                if self._raw_source_path.lower().endswith(".epub"):
                    cover = _extract_cover(self._raw_source_path)
                elif self._raw_source_path.lower().endswith(".pdf"):
                    cover = _extract_pdf_cover(self._raw_source_path)
                if self._should_stop():
                    return
            # Secondary: images the translator has produced in the output
            # folder so far (mid-translation or retranslation runs).
            if not cover:
                cover = _find_cover_in_dir(self._file_path)
                if self._should_stop():
                    return
            # Tertiary: compiled output EPUB (if any) for finished-but-not-
            # organized novels.
            if not cover:
                try:
                    for entry in os.scandir(self._file_path):
                        if self._should_stop():
                            return
                        if (entry.is_file(follow_symlinks=False)
                                and entry.name.lower().endswith(".epub")):
                            cover = _extract_cover(entry.path)
                            if self._should_stop():
                                return
                            if cover:
                                break
                except (PermissionError, OSError):
                    pass
        else:
            cover = _find_folder_cover(self._file_path, config=self._config,
                                       original_path=self._original_path)
            if self._should_stop():
                return
            if not cover:
                for pdf_candidate in (
                    self._file_path,
                    self._raw_source_path,
                    self._original_path,
                ):
                    if (pdf_candidate
                            and str(pdf_candidate).lower().endswith(".pdf")
                            and os.path.isfile(pdf_candidate)):
                        cover = _extract_pdf_cover(pdf_candidate)
                        if cover or self._should_stop():
                            break
        if not self._should_stop():
            self.result_ready.emit(self._file_path, cover or "")


# ---------------------------------------------------------------------------
# Card cover chain for callers without Qt (Glossarion Mobile)
# ---------------------------------------------------------------------------

#: Returned by :func:`resolve_book_cover` when *should_stop* fired mid-chain (the Qt
#: loader emits nothing in that case).
COVER_STOPPED = object()


class _CoverSignal:
    """``result_ready`` stand-in for the plain cover job (records the emit)."""

    def __init__(self):
        self.args = None

    def emit(self, *args):
        self.args = args


class _CoverJob(CoverLoaderMixin):
    def __init__(self, file_path, file_type="epub", config=None, original_path=None,
                 raw_source_path=None, should_stop=None):
        self._file_path = file_path
        self._file_type = file_type
        self._config = config or {}
        self._original_path = original_path
        self._raw_source_path = raw_source_path or ""
        self._cancelled = False
        self._stop_callback = should_stop if callable(should_stop) else None
        self.result_ready = _CoverSignal()

    def _should_stop(self) -> bool:
        if self._cancelled:
            return True
        if self._stop_callback is not None:
            try:
                return bool(self._stop_callback())
            except Exception:
                return False
        return False


def resolve_book_cover(file_path: str, file_type: str = "epub", config: dict | None = None,
                       original_path: str | None = None, raw_source_path: str | None = None,
                       should_stop=None):
    """Cover image path for a Library card (None when nothing was found).

    Runs the desktop card loader (``_CoverLoader.run``, i.e. ``CoverLoaderMixin.run``):
    ``epub`` cards use the EPUB's cover, then the raw EPUB / PDF, an image next to the
    file and :func:`_find_folder_cover`; ``in_progress`` cards (path = output folder)
    the raw source, the folder's images, then a compiled ``.epub`` in it; other kinds
    :func:`_find_folder_cover` then the first image of a PDF. Returns
    :data:`COVER_STOPPED` when *should_stop* fired. Covers are cached in
    ``_cover_cache_dir()`` (see :func:`set_cover_cache_dir`).
    """
    job = _CoverJob(file_path, file_type, config, original_path, raw_source_path, should_stop)
    job.run()
    if job.result_ready.args is None:
        return COVER_STOPPED
    return job.result_ready.args[1] or None


def resolve_card_cover(book: dict, config: dict | None = None, should_stop=None):
    """:func:`resolve_book_cover` for a scanner row (the arguments the desktop card passes)."""
    book = book or {}
    return resolve_book_cover(
        book.get("path", "") or "",
        book.get("type", "epub") or "epub",
        config,
        original_path=book.get("original_path"),
        raw_source_path=book.get("raw_source_path"),
        should_stop=should_stop,
    )


__all__ = [
    "COVER_STOPPED",
    "CoverLoaderMixin",
    "_PDF_COVER_SCAN_PAGE_LIMIT",
    "_cover_cache_dir",
    "_download_remote_cover_image",
    "_extract_cover",
    "_extract_pdf_cover",
    "_find_cover_in_dir",
    "_find_folder_cover",
    "_find_halgakos_icon",
    "_image_bytes_decodable",
    "_probe_image_size",
    "resolve_book_cover",
    "resolve_card_cover",
    "set_cover_cache_dir",
]
