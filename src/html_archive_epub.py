"""Package HTML chapters and chapter ZIPs as EPUBs without changing sources."""

from __future__ import annotations

from dataclasses import dataclass
import html as html_entities
import os
import posixpath
import re
import stat
import tempfile
from typing import Callable
from urllib.parse import quote, unquote, urlsplit
import uuid
import xml.etree.ElementTree as ET
import zipfile

from epub_package import find_epub_opf_member
from image_archive_epub import ImageArchiveConversionCancelled


HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}
RESOURCE_TYPES = {
    ".css": "text/css", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".gif": "image/gif", ".svg": "image/svg+xml",
    ".webp": "image/webp", ".avif": "image/avif", ".bmp": "image/bmp",
    ".ttf": "font/ttf", ".otf": "font/otf", ".woff": "font/woff",
    ".woff2": "font/woff2", ".js": "text/javascript",
    ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".mp4": "video/mp4",
    ".ogg": "audio/ogg", ".ncx": "application/x-dtbncx+xml",
}
XHTML_NS = "http://www.w3.org/1999/xhtml"
OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"
MARKER = b"glossarion:html-archive:v1:"


@dataclass(frozen=True)
class HtmlArchiveEpubResult:
    epub_path: str
    chapter_count: int


def _check_cancelled(should_stop: Callable[[], bool] | None) -> None:
    if should_stop and should_stop():
        raise ImageArchiveConversionCancelled("HTML ZIP conversion cancelled.")


def _natural_key(name: str):
    return [(1, int(part)) if part.isdigit() else (0, part.casefold())
            for part in re.split(r"(\d+)", name)]


def _local_name(tag) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _archive_members(archive, should_stop=None):
    """Accept chapter files and book resources, rejecting ambiguous bundles."""
    members = {}
    seen = set()
    for info in archive.infolist():
        _check_cancelled(should_stop)
        name = info.filename.replace("\\", "/")
        parts = name.split("/")
        if name.startswith("/") or any(part == ".." for part in parts) or ":" in parts[0]:
            raise ValueError(f"Unsafe archive path: {info.filename}")
        name = posixpath.normpath(name)
        if info.is_dir():
            continue
        if (parts[0].casefold() == "__macosx" or parts[-1].startswith("._")
                or parts[-1].casefold() in {".ds_store", "thumbs.db", "desktop.ini"}):
            continue
        if name.casefold() in seen:
            raise ValueError(f"Duplicate archive path: {info.filename}")
        seen.add(name.casefold())
        if info.flag_bits & 1 or stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"Encrypted or linked archive member: {info.filename}")
        ext = posixpath.splitext(name)[1].lower()
        package_resource = (ext == ".opf" or name.casefold() in {
            "mimetype", "meta-inf/container.xml",
        } or name.casefold().endswith("/meta-inf/container.xml"))
        if ext not in HTML_EXTENSIONS and ext not in RESOURCE_TYPES and not package_resource:
            raise ValueError(f"Unsupported member in HTML archive: {info.filename}")
        members[name] = info
    if not any(posixpath.splitext(name)[1].lower() in HTML_EXTENSIONS for name in members):
        raise ValueError("ZIP contains no HTML chapter files.")
    return members


def is_html_archive(path: str, should_stop=None) -> bool:
    """Return whether a ZIP contains HTML chapters and only book resources."""
    try:
        _check_cancelled(should_stop)
        with zipfile.ZipFile(path) as archive:
            _archive_members(archive, should_stop)
        return True
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def _chapter_document(data: bytes, name: str):
    """Keep well-formed source XHTML byte-for-byte; normalize loose HTML."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        root = None
    if root is None or root.tag != f"{{{XHTML_NS}}}html":
        from lxml import etree, html
        from bs4 import UnicodeDammit

        parser = html.HTMLParser(no_network=True)
        decoded = UnicodeDammit(data, is_html=True).unicode_markup
        if decoded is None:
            raise ValueError(f"Cannot decode HTML chapter: {name}")
        decoded = re.sub(r"^\s*<\?xml\b.*?\?>", "", decoded, count=1, flags=re.DOTALL)
        document = html.document_fromstring(decoded, parser=parser)
        for element in document.iter():
            if isinstance(element.tag, str) and not element.tag.startswith("{"):
                element.tag = f"{{{XHTML_NS}}}{element.tag}"
        # A default namespace keeps ordinary HTML tooling able to recognize
        # body/p tags while making the serialized document valid XHTML.
        normalized = etree.Element(f"{{{XHTML_NS}}}html", nsmap={None: XHTML_NS})
        normalized.attrib.update(document.attrib)
        normalized.text = document.text
        normalized.extend(document)
        document = normalized
        head = document.find(f"{{{XHTML_NS}}}head")
        if head is None:
            head = etree.Element(f"{{{XHTML_NS}}}head")
            document.insert(0, head)
        if head.find(f"{{{XHTML_NS}}}title") is None:
            etree.SubElement(head, f"{{{XHTML_NS}}}title").text = posixpath.basename(name)
        data = etree.tostring(document, encoding="utf-8", xml_declaration=True)
        root = ET.fromstring(data)
    title = next(("".join(node.itertext()).strip() for node in root.iter()
                  if _local_name(node.tag) == "title" and "".join(node.itertext()).strip()), "")
    if not title:
        title = next(("".join(node.itertext()).strip() for node in root.iter()
                      if _local_name(node.tag) in {"h1", "h2"}
                      and "".join(node.itertext()).strip()), posixpath.basename(name))
    language = root.attrib.get("lang") or root.attrib.get(
        "{http://www.w3.org/XML/1998/namespace}lang", "und"
    )
    return data, title, language


def _xml(element) -> bytes:
    return ET.tostring(element, encoding="utf-8", xml_declaration=True)


def _container(opf_name: str) -> bytes:
    root = ET.Element("container", {
        "version": "1.0", "xmlns": "urn:oasis:names:tc:opendocument:xmlns:container",
    })
    rootfiles = ET.SubElement(root, "rootfiles")
    ET.SubElement(rootfiles, "rootfile", {
        "full-path": opf_name, "media-type": "application/oebps-package+xml",
    })
    return _xml(root)


def _existing_package(archive, members):
    opf_name = find_epub_opf_member(archive)
    if not opf_name:
        return None
    normalized = posixpath.normpath(opf_name.replace("\\", "/"))
    root = ET.fromstring(archive.read(opf_name))
    if _local_name(root.tag) != "package":
        raise ValueError("Archive OPF is not an EPUB package.")
    manifest = next((node for node in root if _local_name(node.tag) == "manifest"), None)
    spine = next((node for node in root if _local_name(node.tag) == "spine"), None)
    if manifest is None or spine is None:
        raise ValueError("Archive OPF has no manifest or spine.")
    id_to_name = {}
    for item in manifest:
        href = unquote(item.attrib.get("href", "").split("#", 1)[0])
        id_to_name[item.attrib.get("id")] = posixpath.normpath(
            posixpath.join(posixpath.dirname(normalized), href)
        )
    chapter_names = []
    for item in spine:
        if _local_name(item.tag) != "itemref":
            continue
        name = id_to_name.get(item.attrib.get("idref"))
        if name not in members or posixpath.splitext(name)[1].lower() not in HTML_EXTENSIONS:
            raise ValueError("Archive OPF spine references a missing HTML chapter.")
        chapter_names.append(name)
    if not chapter_names:
        raise ValueError("Archive OPF spine contains no HTML chapters.")
    return normalized, chapter_names


def _generated_package(members, chapters, book_title, package_dir):
    opf_name = f"{package_dir}/content.opf"
    uid = "urn:glossarion:html-archive:" + str(uuid.uuid4())
    package = ET.Element("package", {
        "xmlns": OPF_NS, "xmlns:dc": DC_NS, "version": "3.0", "unique-identifier": "book-id",
    })
    metadata = ET.SubElement(package, "metadata")
    ET.SubElement(metadata, "dc:identifier", {"id": "book-id"}).text = uid
    ET.SubElement(metadata, "dc:title").text = book_title
    ET.SubElement(metadata, "dc:language").text = next(iter(chapters.values()))[2]
    from datetime import datetime, timezone
    ET.SubElement(metadata, "meta", {"property": "dcterms:modified"}).text = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    manifest = ET.SubElement(package, "manifest")
    chapter_ids = {}
    for index, name in enumerate(sorted(members, key=_natural_key), 1):
        ext = posixpath.splitext(name)[1].lower()
        if ext not in HTML_EXTENSIONS and ext not in RESOURCE_TYPES:
            continue
        item_id = f"item-{index}"
        attributes = {
            "id": item_id, "href": quote(posixpath.relpath(name, package_dir), safe="/"),
            "media-type": "application/xhtml+xml" if ext in HTML_EXTENSIONS else RESOURCE_TYPES[ext],
        }
        ET.SubElement(manifest, "item", attributes)
        if name in chapters:
            chapter_ids[name] = item_id
    ET.SubElement(manifest, "item", {
        "id": "nav", "href": "nav.xhtml", "media-type": "application/xhtml+xml", "properties": "nav",
    })
    ET.SubElement(manifest, "item", {
        "id": "ncx", "href": "toc.ncx", "media-type": "application/x-dtbncx+xml",
    })
    spine = ET.SubElement(package, "spine", {"toc": "ncx"})
    for name in chapters:
        ET.SubElement(spine, "itemref", {"idref": chapter_ids[name]})
    nav = ET.Element("html", {"xmlns": XHTML_NS, "xmlns:epub": "http://www.idpf.org/2007/ops"})
    ET.SubElement(ET.SubElement(nav, "head"), "title").text = book_title
    nav_list = ET.SubElement(ET.SubElement(ET.SubElement(nav, "body"), "nav", {
        "epub:type": "toc", "id": "toc",
    }), "ol")
    ncx = ET.Element("ncx", {"xmlns": "http://www.daisy.org/z3986/2005/ncx/", "version": "2005-1"})
    ncx_head = ET.SubElement(ncx, "head")
    ET.SubElement(ncx_head, "meta", {"name": "dtb:uid", "content": uid})
    ET.SubElement(ET.SubElement(ncx, "docTitle"), "text").text = book_title
    nav_map = ET.SubElement(ncx, "navMap")
    for index, (name, (_data, title, _lang)) in enumerate(chapters.items(), 1):
        href = quote(posixpath.relpath(name, package_dir), safe="/")
        ET.SubElement(ET.SubElement(nav_list, "li"), "a", {"href": href}).text = title
        point = ET.SubElement(nav_map, "navPoint", {"id": f"chapter-{index}", "playOrder": str(index)})
        ET.SubElement(ET.SubElement(point, "navLabel"), "text").text = title
        ET.SubElement(point, "content", {"src": href})
    return opf_name, {
        opf_name: _xml(package), f"{package_dir}/nav.xhtml": _xml(nav),
        f"{package_dir}/toc.ncx": _xml(ncx),
    }


def convert_html_archive_to_epub(
    archive_path: str, epub_path: str, should_stop=None, *, book_title: str | None = None,
) -> HtmlArchiveEpubResult:
    """Create an EPUB atomically, preserving source paths and existing OPF order."""
    if os.path.normcase(os.path.abspath(archive_path)) == os.path.normcase(os.path.abspath(epub_path)):
        raise ValueError("EPUB output must differ from the source ZIP.")
    _check_cancelled(should_stop)
    output_dir = os.path.dirname(os.path.abspath(epub_path))
    os.makedirs(output_dir, exist_ok=True)
    temporary_path = None
    try:
        with zipfile.ZipFile(archive_path) as source:
            members = _archive_members(source, should_stop)
            existing = _existing_package(source, members)
            chapter_names = existing[1] if existing else sorted(
                (name for name in members if posixpath.splitext(name)[1].lower() in HTML_EXTENSIONS),
                key=_natural_key,
            )
            chapters = {}
            # Include non-spine HTML resources in normalization too, but never
            # add them to an existing authoritative reading order.
            documents = {}
            for name, info in members.items():
                if posixpath.splitext(name)[1].lower() in HTML_EXTENSIONS:
                    _check_cancelled(should_stop)
                    documents[name] = _chapter_document(source.read(info), name)
            for name in chapter_names:
                chapters[name] = documents[name]
            if existing:
                opf_name, _ = existing
                generated = {}
            else:
                package_dir = "_glossarion"
                while any(name.casefold().startswith(package_dir.casefold() + "/") for name in members):
                    package_dir += "_"
                opf_name, generated = _generated_package(
                    members, chapters,
                    book_title or os.path.splitext(os.path.basename(archive_path))[0], package_dir,
                )
            generated["META-INF/container.xml"] = _container(opf_name)
            fd, temporary_path = tempfile.mkstemp(prefix=".html_archive_", suffix=".epub.tmp", dir=output_dir)
            os.close(fd)
            with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as output:
                output.writestr("mimetype", b"application/epub+zip", compress_type=zipfile.ZIP_STORED)
                for name, info in members.items():
                    _check_cancelled(should_stop)
                    if name.casefold() in {"mimetype", "meta-inf/container.xml"}:
                        continue
                    if name in documents:
                        output.writestr(name, documents[name][0])
                    else:
                        with source.open(info) as reader, output.open(name, "w") as writer:
                            while True:
                                _check_cancelled(should_stop)
                                chunk = reader.read(1024 * 1024)
                                if not chunk:
                                    break
                                writer.write(chunk)
                for name, data in generated.items():
                    _check_cancelled(should_stop)
                    output.writestr(name, data)
                output.comment = MARKER + str(len(chapter_names)).encode("ascii")
        _check_cancelled(should_stop)
        os.replace(temporary_path, epub_path)
        temporary_path = None
        return HtmlArchiveEpubResult(str(epub_path), len(chapter_names))
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _css_resource_urls(css: str):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    for match in re.finditer(
        r'''url\(\s*(?:"([^"]*)"|'([^']*)'|([^)]*?))\s*\)''', css, re.IGNORECASE,
    ):
        yield next((value for value in match.groups() if value is not None), "").strip()
    for match in re.finditer(r'''@import\s+["']([^"']+)["']''', css, re.IGNORECASE):
        yield match.group(1)


def _document_resource_urls(data: bytes, extension: str):
    from bs4 import BeautifulSoup, UnicodeDammit

    decoded = UnicodeDammit(data, is_html=extension in HTML_EXTENSIONS).unicode_markup
    if decoded is None:
        return
    if extension == ".css":
        yield from _css_resource_urls(decoded)
        return
    # Resource attributes only: ordinary hyperlinks and iframe chapters are
    # deliberately excluded so selecting one chapter remains one chapter.
    soup = BeautifulSoup(decoded, "html.parser")
    for element in soup.find_all(True):
        name = element.name.rsplit(":", 1)[-1].lower()
        if name in {"img", "source", "audio", "video", "script", "input", "track", "embed"}:
            if element.get("src"):
                yield element["src"]
        if name == "video" and element.get("poster"):
            yield element["poster"]
        if name == "object" and element.get("data"):
            yield element["data"]
        if name in {"link", "image", "use"}:
            reference = element.get("href") or element.get("xlink:href")
            if reference:
                yield reference
        if name in {"img", "source"} and element.get("srcset"):
            for candidate in element["srcset"].split(","):
                if candidate.strip():
                    yield candidate.strip().split()[0]
        if element.get("style"):
            yield from _css_resource_urls(element["style"])
        if name == "style":
            yield from _css_resource_urls(element.get_text())


def _collect_html_file_resources(html_path: str, should_stop=None):
    paths = {os.path.normcase(html_path): html_path}
    pending = [html_path]
    while pending:
        _check_cancelled(should_stop)
        path = pending.pop()
        with open(path, "rb") as stream:
            data = stream.read()
        extension = os.path.splitext(path)[1].lower()
        for reference in _document_resource_urls(data, extension):
            _check_cancelled(should_stop)
            try:
                url = urlsplit(str(reference).strip())
            except ValueError:
                continue
            if url.scheme or url.netloc or not url.path:
                continue
            resource_path = unquote(url.path).replace("\\", os.sep).replace("/", os.sep)
            # Absolute URLs cannot retain their meaning inside an EPUB while
            # preserving the source chapter. Only local relative URLs qualify.
            if os.path.isabs(resource_path):
                continue
            resource_path = os.path.abspath(os.path.join(os.path.dirname(path), resource_path))
            resource_ext = os.path.splitext(resource_path)[1].lower()
            if resource_ext not in RESOURCE_TYPES or not os.path.isfile(resource_path):
                continue
            resource_key = os.path.normcase(resource_path)
            if resource_key in paths:
                continue
            paths[resource_key] = resource_path
            if resource_ext in {".css", ".svg"}:
                pending.append(resource_path)
    return list(paths.values())


def _rewrite_standalone_self_links(data: bytes, html_path: str, chapter_leaf: str) -> bytes:
    """Update only URLs pointing to the selected file when its leaf is aliased."""
    from bs4 import UnicodeDammit

    decoded = UnicodeDammit(data, is_html=True)
    encoding = decoded.original_encoding or "utf-8"

    def replace_reference(match):
        try:
            reference = html_entities.unescape(match.group(3).decode(encoding))
            url = urlsplit(reference)
        except (UnicodeError, ValueError):
            return match.group(0)
        if url.scheme or url.netloc or not url.path:
            return match.group(0)
        local_path = unquote(url.path).replace("/", os.sep).replace("\\", os.sep)
        if os.path.isabs(local_path):
            return match.group(0)
        local_path = os.path.abspath(os.path.join(os.path.dirname(html_path), local_path))
        if os.path.normcase(local_path) != os.path.normcase(html_path):
            return match.group(0)
        new_url_path = posixpath.join(posixpath.dirname(url.path.replace("\\", "/")), chapter_leaf)
        reference = reference.replace(url.path, new_url_path, 1)
        value = html_entities.escape(reference, quote=True).encode(encoding)
        return match.group(1) + match.group(2) + value + match.group(2)

    return re.sub(
        rb'''(\b(?:href|src)\s*=\s*)(["'])(.*?)\2''', replace_reference, data,
        flags=re.IGNORECASE | re.DOTALL,
    )


def convert_html_file_to_epub(html_path: str, epub_path: str, should_stop=None) -> HtmlArchiveEpubResult:
    """Package one selected HTML chapter and its referenced local resources.

    Relative image/style/font paths, including parent-directory references,
    retain their layout. Remote resources and links to other chapters are not
    followed. A temporary source ZIP is removed after success or failure.
    """
    html_path = os.path.abspath(html_path)
    epub_path = os.path.abspath(epub_path)
    if os.path.splitext(html_path)[1].lower() not in HTML_EXTENSIONS:
        raise ValueError("Standalone chapter input must be HTML, HTM, or XHTML.")
    if os.path.normcase(html_path) == os.path.normcase(epub_path):
        raise ValueError("EPUB output must differ from the source HTML file.")
    _check_cancelled(should_stop)
    paths = _collect_html_file_resources(html_path, should_stop)
    from Chapter_Extractor import _is_configured_special_file

    # A selected index/cover/title page is the user's chapter. Give it a
    # regular internal chapter name so EPUB special-file filtering cannot
    # discard it merely because of its original filename.
    chapter_leaf = os.path.basename(html_path)
    if _is_configured_special_file(chapter_leaf):
        chapter_leaf = "chapter0001" + os.path.splitext(chapter_leaf)[1]
    common_root = os.path.commonpath([os.path.dirname(path) for path in paths])
    output_dir = os.path.dirname(epub_path)
    os.makedirs(output_dir, exist_ok=True)
    fd, source_zip = tempfile.mkstemp(prefix=".html_source_", suffix=".zip", dir=output_dir)
    os.close(fd)
    try:
        with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                _check_cancelled(should_stop)
                member_name = os.path.relpath(path, common_root).replace(os.sep, "/")
                if path == html_path and chapter_leaf != os.path.basename(html_path):
                    member_name = posixpath.join(posixpath.dirname(member_name), chapter_leaf)
                    with open(path, "rb") as reader:
                        data = _rewrite_standalone_self_links(reader.read(), html_path, chapter_leaf)
                    archive.writestr(member_name, data)
                    continue
                with open(path, "rb") as reader, archive.open(member_name, "w") as writer:
                    while True:
                        _check_cancelled(should_stop)
                        chunk = reader.read(1024 * 1024)
                        if not chunk:
                            break
                        writer.write(chunk)
        return convert_html_archive_to_epub(
            source_zip, epub_path, should_stop=should_stop,
            book_title=os.path.splitext(os.path.basename(html_path))[0],
        )
    finally:
        os.unlink(source_zip)


def generated_html_epub_chapter_count(path: str) -> int:
    """Return the spine size only for EPUBs created by this converter."""
    try:
        with zipfile.ZipFile(path) as archive:
            if not archive.comment.startswith(MARKER):
                return 0
            count = int(archive.comment[len(MARKER):])
            opf_name = find_epub_opf_member(archive)
            root = ET.fromstring(archive.read(opf_name))
            actual = sum(1 for node in root.iter() if _local_name(node.tag) == "itemref")
            return count if count > 0 and count == actual else 0
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError, zipfile.BadZipFile):
        return 0
