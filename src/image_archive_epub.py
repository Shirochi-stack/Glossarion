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
from typing import Callable
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
    chapter_count: int = 0


class ImageArchiveConversionCancelled(Exception):
    """Raised when the caller requests cancellation during archive conversion."""


def _check_cancelled(should_stop: Callable[[], bool] | None = None) -> None:
    if should_stop and should_stop():
        raise ImageArchiveConversionCancelled("Image ZIP conversion cancelled.")


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


def needs_image_archive_group_rebuild(path: str) -> bool:
    """Return True for older generated image EPUBs with one XHTML per image."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = {_zip_name(name) for name in zf.namelist()}
            if not any(name == "OEBPS/content.opf" for name in names):
                return False
            try:
                opf = zf.read("OEBPS/content.opf").decode("utf-8", errors="ignore")
            except Exception:
                return False
            if "urn:glossarion:image-archive:" not in opf:
                return False
            return any(
                name.startswith("OEBPS/chapters/page_") and name.endswith((".xhtml", ".html"))
                for name in names
            )
    except (OSError, zipfile.BadZipFile):
        return False


def _collect_from_zip(
    zf: zipfile.ZipFile,
    prefix: str,
    depth: int,
    max_depth: int,
    unsupported: list[str],
    nested_count: list[int],
    member_chain: tuple[str, ...] = (),
    should_stop: Callable[[], bool] | None = None,
) -> list[ArchiveImage]:
    _check_cancelled(should_stop)
    images: list[ArchiveImage] = []
    infos = [
        info for info in zf.infolist()
        if not info.is_dir() and not _is_ignored_member(info.filename)
    ]
    infos.sort(key=lambda info: _natural_sort_key(info.filename))

    for info in infos:
        _check_cancelled(should_stop)
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
                _check_cancelled(should_stop)
                nested_count[0] += 1
                nested_bytes = zf.read(info)
                _check_cancelled(should_stop)
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
                            should_stop=should_stop,
                        )
                    )
            except ImageArchiveConversionCancelled:
                raise
            except Exception:
                unsupported.append(display_name)
            continue

        unsupported.append(display_name)

    return images


def collect_image_archive_entries(
    path: str,
    max_depth: int = 4,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[ArchiveImage], tuple[str, ...], int]:
    """Collect image references from an archive, expanding nested ZIP/CBZ files.

    Direct image bytes are not loaded during this scan. Nested archive bytes
    are read only long enough to inspect their central directory.
    """
    unsupported: list[str] = []
    nested_count = [0]
    try:
        _check_cancelled(should_stop)
        with zipfile.ZipFile(path, "r") as zf:
            images = _collect_from_zip(
                zf,
                prefix="",
                depth=0,
                max_depth=max_depth,
                unsupported=unsupported,
                nested_count=nested_count,
                should_stop=should_stop,
            )
    except (OSError, zipfile.BadZipFile) as exc:
        return [], (str(exc),), 0

    return images, tuple(unsupported), nested_count[0]


def _write_image_members(
    source_zf: zipfile.ZipFile,
    output_zf: zipfile.ZipFile,
    pending: list[tuple[tuple[str, ...], str]],
    should_stop: Callable[[], bool] | None = None,
) -> None:
    _check_cancelled(should_stop)
    nested_groups: dict[str, list[tuple[tuple[str, ...], str]]] = {}

    for chain, target_path in pending:
        _check_cancelled(should_stop)
        if not chain:
            continue
        if len(chain) == 1:
            output_zf.writestr(target_path, source_zf.read(chain[0]), compress_type=zipfile.ZIP_STORED)
        else:
            nested_groups.setdefault(chain[0], []).append((chain[1:], target_path))

    for nested_name, group in nested_groups.items():
        _check_cancelled(should_stop)
        nested_bytes = source_zf.read(nested_name)
        _check_cancelled(should_stop)
        with zipfile.ZipFile(io.BytesIO(nested_bytes), "r") as nested_zf:
            _write_image_members(nested_zf, output_zf, group, should_stop=should_stop)


def scan_image_archive(
    path: str,
    max_depth: int = 4,
    should_stop: Callable[[], bool] | None = None,
) -> ImageArchiveScan:
    images, unsupported, nested_count = collect_image_archive_entries(
        path,
        max_depth=max_depth,
        should_stop=should_stop,
    )
    return ImageArchiveScan(
        is_image_archive=bool(images) and not unsupported,
        image_count=len(images),
        nested_archive_count=nested_count,
        unsupported_entries=unsupported,
    )


def _image_groups(images: list[ArchiveImage], nested_count: int) -> list[tuple[str, list[ArchiveImage]]]:
    """Return chapter groups for the generated EPUB.

    Nested ZIP bundles usually represent chapters/episodes/volumes. Grouping
    every image from one inner ZIP into one XHTML file keeps Glossarion's
    extractor from having to parse tens of thousands of one-image chapters,
    while still letting the existing image rename pass assign
    chapterNNNN_img_M names.
    """
    if not nested_count:
        return [(f"Page {idx}", [image]) for idx, image in enumerate(images, 1)]

    grouped: list[tuple[str, list[ArchiveImage]]] = []
    group_index: dict[str, int] = {}
    for image in images:
        key = image.member_chain[0] if len(image.member_chain) > 1 else "__loose_images__"
        if key not in group_index:
            if key == "__loose_images__":
                title = "Loose Images"
            else:
                title = os.path.splitext(os.path.basename(_zip_name(key)))[0] or _zip_name(key)
            group_index[key] = len(grouped)
            grouped.append((title, []))
        grouped[group_index[key]][1].append(image)
    return grouped


def _chapter_xhtml(title: str, image_hrefs: list[str], chapter_num: int) -> bytes:
    title_xml = _safe_xml_text(title)
    image_tags = []
    for idx, image_href in enumerate(image_hrefs, 1):
        image_href_xml = _safe_xml_text(image_href)
        image_tags.append(
            f'    <img src="{image_href_xml}" alt="Image {idx}"/>\n'
        )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" lang="und" xml:lang="und">\n'
        f"<head><title>{title_xml}</title><link rel=\"stylesheet\" type=\"text/css\" href=\"../styles.css\"/></head>\n"
        '<body>\n'
        f'  <section class="image-page" id="chapter-{chapter_num:04d}">\n'
        + "".join(image_tags) +
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
    should_stop: Callable[[], bool] | None = None,
) -> ImageArchiveEpubResult:
    """Convert an image-only ZIP/CBZ archive into a valid EPUB package."""
    if not archive_path or not os.path.isfile(archive_path):
        raise FileNotFoundError(archive_path)

    _check_cancelled(should_stop)
    images, unsupported, nested_count = collect_image_archive_entries(
        archive_path,
        max_depth=max_depth,
        should_stop=should_stop,
    )
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
    groups = _image_groups(images, nested_count)
    image_idx = 0

    for chapter_idx, (group_title, group_images) in enumerate(groups, 1):
        _check_cancelled(should_stop)
        chapter_href = f"chapters/chapter_{chapter_idx:04d}.xhtml"
        chapter_items.append((f"chapter_{chapter_idx:04d}", chapter_href))
        chapter_image_hrefs: list[str] = []

        for image_in_chapter_idx, image in enumerate(group_images, 1):
            _check_cancelled(should_stop)
            image_idx += 1
            ext = image.extension if image.extension in IMAGE_EXTS else ".jpg"
            stem = _safe_internal_stem(image.source_name, f"image_{image_in_chapter_idx:04d}")
            image_name = f"chapter_{chapter_idx:04d}_src_{image_in_chapter_idx:04d}_{stem}{ext}"
            image_href = f"images/{image_name}"
            image_items.append((f"img_{image_idx:05d}", image_href, image.media_type))
            image_payloads.append((image.member_chain, f"OEBPS/{image_href}"))
            chapter_image_hrefs.append(f"../{image_href}")

        chapter_payloads.append(
            (
                f"OEBPS/{chapter_href}",
                _chapter_xhtml(group_title or f"Chapter {chapter_idx}", chapter_image_hrefs, chapter_idx),
            )
        )

    tmp_path = f"{epub_path}.tmp"
    try:
        with zipfile.ZipFile(tmp_path, "w") as zf:
            _check_cancelled(should_stop)
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
                _write_image_members(source_zf, zf, image_payloads, should_stop=should_stop)
            for name, data in chapter_payloads:
                _check_cancelled(should_stop)
                zf.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)

        _check_cancelled(should_stop)
        os.replace(tmp_path, epub_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise
    return ImageArchiveEpubResult(
        epub_path=epub_path,
        image_count=len(images),
        nested_archive_count=nested_count,
        ignored_entry_count=len(unsupported) if allow_unsupported else 0,
        chapter_count=len(groups),
    )
