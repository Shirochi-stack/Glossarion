"""Source HTML used to compile EPUBs in Image output mode."""

import json
import os
import posixpath
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree as ET

from epub_package import find_epub_opf_member
from html_output_utils import ensure_utf8_html_document, write_utf8_html_file


RAW_HTML_DIR = "RAW"
HTML_EXTENSIONS = (".html", ".htm", ".xhtml")
_IMAGE_ATTRIBUTES = {
    "img": ("src", "srcset", "data-src"),
    "image": ("href", "xlink:href"),
    "source": ("srcset",),
    "object": ("data",),
    "video": ("poster",),
}
_IMAGE_MARKUP = re.compile(r"<(?:[\w.-]+:)?(?:img|image|svg|source|object|video)\b|\burl\s*\(", re.I)
_BACKGROUND_IMAGE = re.compile(r"\bbackground(?:-image)?\s*:[^;{}]*\burl\s*\(", re.I)


def _has_image_markup(markup):
    if not _IMAGE_MARKUP.search(markup):
        return False
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(markup, "html.parser")
    for tag in soup.find_all(True):
        name = tag.name.rsplit(":", 1)[-1]
        if name == "svg" or any(tag.get(attr) for attr in _IMAGE_ATTRIBUTES.get(name, ())):
            return True
        style = tag.get_text() if name == "style" else tag.get("style") or ""
        if _BACKGROUND_IMAGE.search(style):
            return True
    return False


def inspect_epub_image_chapters(source, spine_chapters):
    """Find image-free spine rows and the original image names of cover pages."""
    no_images = set()
    covers = {}
    try:
        with zipfile.ZipFile(source) as archive:
            opf = find_epub_opf_member(archive)
            if not opf:
                return no_images, covers
            package = ET.fromstring(archive.read(opf))
            cover_ids = set()
            cover_pages = set()
            manifest = {}
            for element in package.iter():
                tag = element.tag.rsplit("}", 1)[-1]
                href = unquote(element.get("href", "").split("#", 1)[0])
                if tag == "item":
                    manifest[element.get("id")] = href
                    if "cover-image" in element.get("properties", "").split():
                        cover_ids.add(element.get("id"))
                elif tag == "meta" and element.get("name", "").casefold() == "cover":
                    cover_ids.add(element.get("content"))
                elif tag == "reference" and "cover" in element.get("type", "").casefold().split():
                    cover_pages.add(posixpath.normpath(href))
            cover_images = {
                posixpath.basename(manifest[item])
                for item in cover_ids if manifest.get(item)
            }
            base = posixpath.dirname(opf)
            members = set(archive.namelist())
            for chapter in spine_chapters:
                href = chapter.get("href")
                if not href:
                    continue
                member = posixpath.normpath(posixpath.join(base, href.replace("\\", "/"))).lstrip("/")
                if member not in members:
                    # Hrefs can include anchors. Try the literal name first
                    # because the OPF reader has already decoded %23.
                    member = re.split(r"[?#]", member, maxsplit=1)[0]
                try:
                    markup = decode_source_html(archive.read(member))
                except (KeyError, OSError, RuntimeError, zipfile.BadZipFile):
                    continue
                if not markup:
                    continue
                position = chapter["position"]
                if not _has_image_markup(markup):
                    no_images.add(position)
                    continue
                stem = posixpath.splitext(posixpath.basename(href))[0].casefold()
                named_cover = re.fullmatch(r"(?:p[-_])?cover(?:[-_].*|page)?", stem)
                declared_page = posixpath.normpath(href.split("#", 1)[0]) in cover_pages
                if declared_page or named_cover or (position == 0 and cover_images):
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(markup, "html.parser")
                    referenced = {
                        posixpath.basename(unquote(value.split("#", 1)[0].split("?", 1)[0]))
                        for tag in soup.find_all(True)
                        for attr in _IMAGE_ATTRIBUTES.get(tag.name.rsplit(":", 1)[-1], ())
                        if attr != "srcset"
                        and isinstance(value := tag.get(attr), str)
                        and value and not value.startswith("data:")
                    }
                    matches_cover = bool(
                        {name.casefold() for name in referenced} & {name.casefold() for name in cover_images}
                    )
                    if declared_page or named_cover or matches_cover:
                        covers[position] = referenced or cover_images
    except (OSError, KeyError, ET.ParseError, zipfile.BadZipFile):
        pass
    return no_images, covers


def load_image_rename_map(output_dir):
    try:
        data = json.loads((Path(output_dir) / "image_rename_map.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def decode_source_html(raw_bytes):
    for encoding in ("utf-8", "utf-16", "gb18030", "shift_jis", "euc-kr"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return ""


def prepare_source_html(markup, rename_map):
    if markup and rename_map:
        from bs4 import BeautifulSoup
        from image_reference_map import apply_filename_map

        soup = BeautifulSoup(markup, "html.parser")
        if apply_filename_map(soup, rename_map):
            return str(soup)
    return markup


def list_chapter_html(directory):
    try:
        with os.scandir(directory) as entries:
            return [
                entry.name for entry in entries
                if entry.name.lower().endswith(HTML_EXTENSIONS) and entry.is_file()
            ]
    except FileNotFoundError:
        return []


def write_epub_image_html(source, output_dir, chapters, check_stop):
    """Copy chapter markup under RAW/ using stable source filenames."""
    raw_dir = Path(output_dir) / RAW_HTML_DIR
    rename_map = load_image_rename_map(output_dir)
    copied = 0
    with zipfile.ZipFile(source) as archive:
        for chapter in chapters:
            if check_stop():
                return None
            member = chapter["filename"]
            try:
                markup = decode_source_html(archive.read(member))
            except KeyError:
                markup = ""
            if not markup:
                markup = (
                    chapter.get("original_html") or chapter.get("source_html")
                    or chapter.get("raw_html") or chapter.get("body") or ""
                )
            write_utf8_html_file(raw_dir / Path(member).name, prepare_source_html(markup, rename_map))
            copied += 1
    return copied


def migrate_legacy_image_html(source, output_dir, progress):
    """Move only old image-mode output that still matches its source markup."""
    if (
        not str(source or "").lower().endswith(".epub")
        or not isinstance(progress, dict)
        or progress.get("epub_image_html_in_raw")
    ):
        return 0

    image_gen = progress.get("image_gen") or {}
    last_image_run = image_gen.get("last_updated") if isinstance(image_gen, dict) else None
    was_image_mode = progress.get("output_mode") == "image"
    if not was_image_mode and not isinstance(last_image_run, (int, float)):
        return 0

    chapter_rows = progress.get("chapters")
    if not isinstance(chapter_rows, dict):
        return 0
    chapters = [row for row in chapter_rows.values() if isinstance(row, dict)]
    if any(row.get("status") == "in_progress" for row in chapters):
        return 0

    output_dir = Path(output_dir)
    candidates = {}
    for row in chapters:
        name = str(row.get("output_file") or "")
        if (
            not name.lower().endswith(HTML_EXTENSIONS)
            or "/" in name or "\\" in name or ":" in name
            or row.get("status") != "completed"
        ):
            continue
        timestamp = row.get("last_updated")
        if not was_image_mode and (
            not isinstance(timestamp, (int, float)) or timestamp > last_image_run
        ):
            continue
        if (output_dir / name).is_file():
            candidates[name] = str(row.get("original_basename") or name)
    if not candidates:
        return 0

    rename_map = load_image_rename_map(output_dir)
    raw_dir = output_dir / RAW_HTML_DIR
    moved = 0

    def key(name):
        name = name.rsplit("/", 1)[-1].casefold().removeprefix("response_")
        while name.endswith((*HTML_EXTENSIONS, ".xml")):
            name = os.path.splitext(name)[0]
        return name

    def normalized(markup):
        return ensure_utf8_html_document(markup).replace("\r\n", "\n").replace("\r", "\n")

    try:
        with zipfile.ZipFile(source) as archive:
            members = {}
            for member in archive.namelist():
                if member.lower().endswith(HTML_EXTENSIONS):
                    members.setdefault(key(member), []).append(member)
            for name, original in candidates.items():
                matches = members.get(key(original), [])
                if len(matches) != 1:
                    continue
                path = output_dir / name
                destination = raw_dir / Path(matches[0]).name
                try:
                    markup = decode_source_html(archive.read(matches[0]))
                    if not markup:
                        continue
                    existing = path.read_text(encoding="utf-8")
                    if existing != normalized(markup) and existing != normalized(
                        prepare_source_html(markup, rename_map)
                    ):
                        continue
                    raw_dir.mkdir(exist_ok=True)
                    if destination.exists():
                        # A RAW copy already exists. Only remove this duplicate
                        # if that copy also preserves the same source markup.
                        if destination.read_text(encoding="utf-8") not in (
                            existing, normalized(prepare_source_html(markup, rename_map))
                        ):
                            continue
                        path.unlink()
                    else:
                        path.replace(destination)
                    moved += 1
                except (OSError, UnicodeError, zipfile.BadZipFile):
                    continue
    except (OSError, zipfile.BadZipFile):
        return moved
    return moved
