"""Build valid EPUB files from image-only ZIP/CBZ archives."""

from __future__ import annotations

from dataclasses import dataclass
import html
import hashlib
import io
import mimetypes
import os
import posixpath
import re
import time
import zipfile


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".avif")
NESTED_ARCHIVE_EXTS = (".zip", ".cbz")
IGNORED_FILENAMES = {
    ".ds_store",
    "thumbs.db",
    "desktop.ini",
    "comicinfo.xml",
}


@dataclass(frozen=True)
class ArchiveImage:
    source_name: str
    extension: str
    media_type: str
    member_chain: tuple[str, ...]


@dataclass(frozen=True)
class ImageArchiveScan:
    is_image_archive: bool
    image_count: int
    nested_archive_count: int
    unsupported_entries: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImageArchiveEpubResult:
    epub_path: str
    image_count: int
    nested_archive_count: int
    ignored_entry_count: int = 0


def _natural_sort_key(value: str):
    parts = re.split(r"(\d+)", str(value).replace("\\", "/"))
    return [int(part) if part.isdigit() else part.casefold() for part in parts]


def _zip_name(name: str) -> str:
    clean = posixpath.normpath(str(name or "").replace("\\", "/"))
    while clean.startswith("./"):
        clean = clean[2:]
    clean = clean.lstrip("/")
    return "" if clean == "." else clean


def _is_ignored_member(name: str) -> bool:
    clean = _zip_name(name)
    if not clean or clean.endswith("/"):
        return True
    parts = [p for p in clean.split("/") if p]
    if not parts:
        return True
    if parts[0].casefold() == "__macosx":
        return True
    base = parts[-1]
    return base.casefold() in IGNORED_FILENAMES or base.startswith("._")


def _safe_xml_text(value: str) -> str:
    return html.escape(str(value or ""), quote=True)


def _safe_internal_stem(value: str, fallback: str) -> str:
    stem = os.path.splitext(os.path.basename(str(value or "")))[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem[:80] or fallback


def _media_type_for(ext: str) -> str:
    ext = (ext or "").lower()
    if ext == ".jpg":
        return "image/jpeg"
    guessed = mimetypes.types_map.get(ext)
    return guessed or f"image/{ext.lstrip('.') or 'jpeg'}"


def is_epub_zip(path: str) -> bool:
    """Return True when a ZIP-shaped file already contains EPUB structure."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = {_zip_name(name).casefold() for name in zf.namelist()}
            if "meta-inf/container.xml" in names and any(name.endswith(".opf") for name in names):
                return True
            if "mimetype" in names:
                try:
                    return zf.read("mimetype").decode("ascii", errors="ignore").strip() == "application/epub+zip"
                except Exception:
                    return False
    except (OSError, zipfile.BadZipFile):
        return False
    return False


def _collect_from_zip(
    zf: zipfile.ZipFile,
    prefix: str,
    depth: int,
    max_depth: int,
    unsupported: list[str],
    nested_count: list[int],
    member_chain: tuple[str, ...] = (),
) -> list[ArchiveImage]:
    images: list[ArchiveImage] = []
    infos = [
        info for info in zf.infolist()
        if not info.is_dir() and not _is_ignored_member(info.filename)
    ]
    infos.sort(key=lambda info: _natural_sort_key(info.filename))

    for info in infos:
        clean_name = _zip_name(info.filename)
        display_name = f"{prefix}/{clean_name}" if prefix else clean_name
        ext = os.path.splitext(clean_name)[1].lower()

        if ext in IMAGE_EXTS:
            images.append(
                ArchiveImage(
                    source_name=display_name,
                    extension=ext,
                    media_type=_media_type_for(ext),
                    member_chain=(*member_chain, info.filename),
                )
            )
            continue

        if ext in NESTED_ARCHIVE_EXTS:
            if depth >= max_depth:
                unsupported.append(display_name)
                continue
            try:
                nested_count[0] += 1
                nested_bytes = zf.read(info)
                with zipfile.ZipFile(io.BytesIO(nested_bytes), "r") as nested_zf:
                    images.extend(
                        _collect_from_zip(
                            nested_zf,
                            prefix=display_name,
                            depth=depth + 1,
                            max_depth=max_depth,
                            unsupported=unsupported,
                            nested_count=nested_count,
                            member_chain=(*member_chain, info.filename),
                        )
                    )
            except Exception:
                unsupported.append(display_name)
            continue

        unsupported.append(display_name)

    return images


def collect_image_archive_entries(path: str, max_depth: int = 4) -> tuple[list[ArchiveImage], tuple[str, ...], int]:
    """Collect image references from an archive, expanding nested ZIP/CBZ files.

    Direct image bytes are not loaded during this scan. Nested archive bytes
    are read only long enough to inspect their central directory.
    """
    unsupported: list[str] = []
    nested_count = [0]
    try:
        with zipfile.ZipFile(path, "r") as zf:
            images = _collect_from_zip(
                zf,
                prefix="",
                depth=0,
                max_depth=max_depth,
                unsupported=unsupported,
                nested_count=nested_count,
            )
    except (OSError, zipfile.BadZipFile) as exc:
        return [], (str(exc),), 0

    return images, tuple(unsupported), nested_count[0]


def _write_image_members(
    source_zf: zipfile.ZipFile,
    output_zf: zipfile.ZipFile,
    pending: list[tuple[tuple[str, ...], str]],
) -> None:
    nested_groups: dict[str, list[tuple[tuple[str, ...], str]]] = {}

    for chain, target_path in pending:
        if not chain:
            continue
        if len(chain) == 1:
            output_zf.writestr(target_path, source_zf.read(chain[0]), compress_type=zipfile.ZIP_DEFLATED)
        else:
            nested_groups.setdefault(chain[0], []).append((chain[1:], target_path))

    for nested_name, group in nested_groups.items():
        nested_bytes = source_zf.read(nested_name)
        with zipfile.ZipFile(io.BytesIO(nested_bytes), "r") as nested_zf:
            _write_image_members(nested_zf, output_zf, group)


def scan_image_archive(path: str, max_depth: int = 4) -> ImageArchiveScan:
    images, unsupported, nested_count = collect_image_archive_entries(path, max_depth=max_depth)
    return ImageArchiveScan(
        is_image_archive=bool(images) and not unsupported,
        image_count=len(images),
        nested_archive_count=nested_count,
        unsupported_entries=unsupported,
    )


def _chapter_xhtml(title: str, image_href: str, page_num: int) -> bytes:
    title_xml = _safe_xml_text(title)
    image_href_xml = _safe_xml_text(image_href)
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" lang="und" xml:lang="und">\n'
        f"<head><title>{title_xml}</title><link rel=\"stylesheet\" type=\"text/css\" href=\"../styles.css\"/></head>\n"
        '<body>\n'
        f'  <section class="image-page" id="page-{page_num:04d}">\n'
        f'    <img src="{image_href_xml}" alt="Page {page_num}"/>\n'
        "  </section>\n"
        "</body>\n"
        "</html>\n"
    ).encode("utf-8")


def _content_opf(book_title: str, image_items: list[tuple[str, str, str]], chapter_items: list[tuple[str, str]]) -> bytes:
    title_xml = _safe_xml_text(book_title)
    identifier = hashlib.sha1(book_title.encode("utf-8", errors="ignore")).hexdigest()
    modified = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest = [
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        '<item id="css" href="styles.css" media-type="text/css"/>',
    ]
    for item_id, href, media_type in image_items:
        props = ' properties="cover-image"' if item_id == "img_0001" else ""
        manifest.append(
            f'<item id="{item_id}" href="{_safe_xml_text(href)}" media-type="{_safe_xml_text(media_type)}"{props}/>'
        )
    for item_id, href in chapter_items:
        manifest.append(
            f'<item id="{item_id}" href="{_safe_xml_text(href)}" media-type="application/xhtml+xml"/>'
        )

    spine = [f'<itemref idref="{item_id}"/>' for item_id, _href in chapter_items]
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="bookid" version="3.0">\n'
        "  <metadata xmlns:dc=\"http://purl.org/dc/elements/1.1/\">\n"
        f"    <dc:identifier id=\"bookid\">urn:glossarion:image-archive:{identifier}</dc:identifier>\n"
        f"    <dc:title>{title_xml}</dc:title>\n"
        "    <dc:language>und</dc:language>\n"
        f"    <meta property=\"dcterms:modified\">{modified}</meta>\n"
        "  </metadata>\n"
        "  <manifest>\n    "
        + "\n    ".join(manifest)
        + "\n  </manifest>\n"
        "  <spine toc=\"ncx\">\n    "
        + "\n    ".join(spine)
        + "\n  </spine>\n"
        "</package>\n"
    ).encode("utf-8")


def _nav_xhtml(book_title: str, chapter_items: list[tuple[str, str]]) -> bytes:
    title_xml = _safe_xml_text(book_title)
    links = [
        f'<li><a href="{_safe_xml_text(href)}">Page {idx}</a></li>'
        for idx, (_item_id, href) in enumerate(chapter_items, 1)
    ]
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" lang="und" xml:lang="und">\n'
        f"<head><title>{title_xml}</title></head>\n"
        "<body><nav epub:type=\"toc\" xmlns:epub=\"http://www.idpf.org/2007/ops\"><h1>"
        f"{title_xml}</h1><ol>"
        + "".join(links)
        + "</ol></nav></body>\n"
        "</html>\n"
    ).encode("utf-8")


def _toc_ncx(book_title: str, chapter_items: list[tuple[str, str]]) -> bytes:
    title_xml = _safe_xml_text(book_title)
    nav_points = []
    for idx, (_item_id, href) in enumerate(chapter_items, 1):
        nav_points.append(
            f'<navPoint id="navPoint-{idx}" playOrder="{idx}">'
            f"<navLabel><text>Page {idx}</text></navLabel>"
            f'<content src="{_safe_xml_text(href)}"/>'
            "</navPoint>"
        )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">\n'
        "  <head><meta name=\"dtb:uid\" content=\"glossarion-image-archive\"/></head>\n"
        f"  <docTitle><text>{title_xml}</text></docTitle>\n"
        "  <navMap>"
        + "".join(nav_points)
        + "</navMap>\n"
        "</ncx>\n"
    ).encode("utf-8")


def convert_image_archive_to_epub(
    archive_path: str,
    epub_path: str | None = None,
    title: str | None = None,
    max_depth: int = 4,
    allow_unsupported: bool = False,
) -> ImageArchiveEpubResult:
    """Convert an image-only ZIP/CBZ archive into a valid EPUB package."""
    if not archive_path or not os.path.isfile(archive_path):
        raise FileNotFoundError(archive_path)

    images, unsupported, nested_count = collect_image_archive_entries(archive_path, max_depth=max_depth)
    if not images:
        raise ValueError("Archive does not contain any supported images.")
    if unsupported and not allow_unsupported:
        preview = ", ".join(unsupported[:5])
        extra = f" and {len(unsupported) - 5} more" if len(unsupported) > 5 else ""
        raise ValueError(f"Archive contains non-image entries: {preview}{extra}")

    epub_path = epub_path or os.path.splitext(archive_path)[0] + ".epub"
    os.makedirs(os.path.dirname(os.path.abspath(epub_path)), exist_ok=True)
    book_title = title or os.path.splitext(os.path.basename(archive_path))[0]

    image_items: list[tuple[str, str, str]] = []
    chapter_items: list[tuple[str, str]] = []
    image_payloads: list[tuple[tuple[str, ...], str]] = []
    chapter_payloads: list[tuple[str, bytes]] = []

    for idx, image in enumerate(images, 1):
        ext = image.extension if image.extension in IMAGE_EXTS else ".jpg"
        stem = _safe_internal_stem(image.source_name, f"page_{idx:04d}")
        image_name = f"page_{idx:04d}_{stem}{ext}"
        image_href = f"images/{image_name}"
        chapter_href = f"chapters/page_{idx:04d}.xhtml"
        image_items.append((f"img_{idx:04d}", image_href, image.media_type))
        chapter_items.append((f"page_{idx:04d}", chapter_href))
        image_payloads.append((image.member_chain, f"OEBPS/{image_href}"))
        chapter_payloads.append(
            (
                f"OEBPS/{chapter_href}",
                _chapter_xhtml(f"Page {idx}", f"../{image_href}", idx),
            )
        )

    tmp_path = f"{epub_path}.tmp"
    with zipfile.ZipFile(tmp_path, "w") as zf:
        zf.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "META-INF/container.xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
                '  <rootfiles><rootfile full-path="OEBPS/content.opf" '
                'media-type="application/oebps-package+xml"/></rootfiles>\n'
                "</container>\n"
            ),
            compress_type=zipfile.ZIP_DEFLATED,
        )
        zf.writestr(
            "OEBPS/styles.css",
            (
                "html, body { margin: 0; padding: 0; }\n"
                "body { background: #fff; }\n"
                ".image-page { margin: 0; padding: 0; text-align: center; page-break-after: always; }\n"
                ".image-page img { max-width: 100%; height: auto; display: block; margin: 0 auto; }\n"
            ),
            compress_type=zipfile.ZIP_DEFLATED,
        )
        zf.writestr("OEBPS/content.opf", _content_opf(book_title, image_items, chapter_items), compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/nav.xhtml", _nav_xhtml(book_title, chapter_items), compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/toc.ncx", _toc_ncx(book_title, chapter_items), compress_type=zipfile.ZIP_DEFLATED)

        with zipfile.ZipFile(archive_path, "r") as source_zf:
            _write_image_members(source_zf, zf, image_payloads)
        for name, data in chapter_payloads:
            zf.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)

    os.replace(tmp_path, epub_path)
    return ImageArchiveEpubResult(
        epub_path=epub_path,
        image_count=len(images),
        nested_archive_count=nested_count,
        ignored_entry_count=len(unsupported) if allow_unsupported else 0,
    )
