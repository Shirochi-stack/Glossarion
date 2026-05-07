import html
import os
import posixpath
import re
from urllib.parse import unquote
import zipfile
from typing import Dict, Optional

from bs4 import BeautifulSoup


HTML_EXTS = (".html", ".xhtml", ".htm")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".avif")


def ocr_epub_path_for(source_epub_path: str) -> str:
    """Return the sibling OCR-source EPUB path for an input EPUB."""
    base_dir = os.path.dirname(os.path.abspath(source_epub_path))
    stem, ext = os.path.splitext(os.path.basename(source_epub_path))
    return os.path.join(base_dir, f"{stem}_OCR{ext or '.epub'}")


def _copy_zip_info(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    copied = zipfile.ZipInfo(info.filename, info.date_time)
    copied.comment = info.comment
    copied.extra = info.extra
    copied.internal_attr = info.internal_attr
    copied.external_attr = info.external_attr
    copied.create_system = info.create_system
    copied.compress_type = info.compress_type
    return copied


IMG_TAG_RE = re.compile(r"(<img\b[^>]*?/?>)", re.IGNORECASE | re.DOTALL)


def _text_to_xhtml(text: str, title: str = "") -> bytes:
    title = title or "OCR Source"
    escaped_title = html.escape(title, quote=True)
    escaped_text = html.escape(text or "", quote=False)
    doc = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" '
        '"http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">\n'
        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
        '<head><title>{title}</title></head>\n'
        '<body><pre style="white-space: pre-wrap;">{text}</pre></body>\n'
        '</html>\n'
    ).format(title=escaped_title, text=escaped_text)
    return doc.encode("utf-8")


def _reverse_rename_map(image_rename_map) -> Dict[str, str]:
    reverse = {}
    for old_name, new_name in (image_rename_map or {}).items():
        old_base = os.path.basename(str(old_name or ""))
        new_base = os.path.basename(str(new_name or ""))
        if old_base and new_base:
            reverse[new_base] = old_base
            reverse[new_base.lower()] = old_base
    return reverse


def _source_src_for_marker(src: str, reverse_rename_map: Dict[str, str]) -> str:
    src_text = str(src or "")
    if not src_text or not reverse_rename_map:
        return src_text
    clean_src, suffix = src_text, ""
    for sep in ("?", "#"):
        if sep in clean_src:
            clean_src, tail = clean_src.split(sep, 1)
            suffix = sep + tail
            break
    normalized = clean_src.replace("\\", "/")
    dirname = posixpath.dirname(normalized)
    basename = posixpath.basename(normalized)
    source_base = reverse_rename_map.get(basename) or reverse_rename_map.get(basename.lower())
    if not source_base:
        return src_text
    rewritten = posixpath.join(dirname, source_base) if dirname else source_base
    return rewritten + suffix


def _append_ocr_fragment(soup: BeautifulSoup, parent, text: str, reverse_rename_map=None) -> None:
    """Append OCR text while allowing preserved No-image <img> markers through."""
    reverse_rename_map = reverse_rename_map or {}
    for part in IMG_TAG_RE.split(text or ""):
        if not part:
            continue
        if IMG_TAG_RE.fullmatch(part.strip()):
            img_fragment = BeautifulSoup(part, "html.parser")
            img = img_fragment.find("img")
            if img:
                src = img.get("src", "")
                rewritten_src = _source_src_for_marker(src, reverse_rename_map)
                if rewritten_src and rewritten_src != src:
                    img["src"] = rewritten_src
                parent.append(img)
            continue

        pre = soup.new_tag("pre")
        pre["style"] = "white-space: pre-wrap;"
        pre.string = part.strip()
        if pre.string:
            parent.append(pre)


def _html_with_ocr_body(original_data: bytes, replacement: str, title: str = "", reverse_rename_map=None) -> bytes:
    """Preserve original HTML head/CSS and replace only the visible body content."""
    try:
        original_html = original_data.decode("utf-8", errors="replace")
    except Exception:
        original_html = str(original_data)

    soup = BeautifulSoup(original_html, "html.parser")
    if soup.find("html") is None:
        html_tag = soup.new_tag("html")
        html_tag["xmlns"] = "http://www.w3.org/1999/xhtml"
        html_tag.extend(list(soup.contents))
        soup.append(html_tag)

    html_tag = soup.find("html") or soup
    head = soup.find("head")
    if head is None:
        head = soup.new_tag("head")
        html_tag.insert(0, head)
    if head.find("title") is None:
        title_tag = soup.new_tag("title")
        title_tag.string = title or "OCR Source"
        head.append(title_tag)

    body = soup.find("body")
    if body is None:
        body = soup.new_tag("body")
        html_tag.append(body)
    body.clear()
    _append_ocr_fragment(soup, body, replacement or "", reverse_rename_map)

    return str(soup).encode("utf-8")


def _lookup_replacement(filename: str, replacements: Dict[str, str]) -> Optional[str]:
    if not filename:
        return None
    candidates = []
    normalized = filename.replace("\\", "/")
    base = os.path.basename(normalized)
    stem = os.path.splitext(base)[0].lower()
    candidates.extend([normalized, base, stem])
    for key in candidates:
        if key in replacements:
            return replacements[key]
        low = str(key).lower()
        if low in replacements:
            return replacements[low]
    return None


def _zip_norm(path: str) -> str:
    return posixpath.normpath(unquote(str(path or "").replace("\\", "/"))).lstrip("./")


def _resolve_zip_ref(chapter_zip_path: str, src: str) -> str:
    clean_src = str(src or "").split("?", 1)[0].split("#", 1)[0].replace("\\", "/")
    if not clean_src:
        return ""
    if clean_src.startswith("/"):
        return _zip_norm(clean_src.lstrip("/"))
    base_dir = posixpath.dirname(str(chapter_zip_path or "").replace("\\", "/"))
    return _zip_norm(posixpath.join(base_dir, clean_src) if base_dir else clean_src)


def _is_preserved_original_html(filename: str) -> bool:
    stem = os.path.splitext(os.path.basename(str(filename or "")))[0].lower()
    return stem == "cover" or stem.startswith("cover_") or stem.startswith("cover-")


def _html_image_refs(chapter_zip_path: str, html_data: bytes) -> set:
    refs = set()
    try:
        source = html_data.decode("utf-8", errors="replace")
    except Exception:
        source = str(html_data)
    try:
        soup = BeautifulSoup(source, "html.parser")
        for img in soup.find_all("img"):
            resolved = _resolve_zip_ref(chapter_zip_path, img.get("src", ""))
            if resolved:
                refs.add(resolved)
                refs.add(_zip_norm(img.get("src", "")))
    except Exception:
        pass
    return refs


def _replacement_image_refs(chapter_zip_path: str, replacement: str, reverse_rename_map=None) -> set:
    refs = set()
    reverse_rename_map = reverse_rename_map or {}
    for marker in IMG_TAG_RE.findall(replacement or ""):
        try:
            soup = BeautifulSoup(marker, "html.parser")
            img = soup.find("img")
            src = img.get("src") if img else ""
            src = _source_src_for_marker(src, reverse_rename_map)
            resolved = _resolve_zip_ref(chapter_zip_path, src)
            if resolved:
                refs.add(resolved)
                refs.add(_zip_norm(src))
        except Exception:
            continue
    return refs


def write_ocr_epub(
    source_epub_path: str,
    chapter_text_by_filename: Dict[str, str],
    output_path: Optional[str] = None,
    image_rename_map: Optional[Dict[str, str]] = None,
) -> str:
    """Copy an EPUB and replace chapter HTML bodies with raw OCR/text content."""
    if not source_epub_path or not os.path.exists(source_epub_path):
        raise FileNotFoundError(source_epub_path)
    output_path = output_path or ocr_epub_path_for(source_epub_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    replacements = {}
    reverse_rename_map = _reverse_rename_map(image_rename_map)
    for key, value in (chapter_text_by_filename or {}).items():
        if not key or not str(value).strip():
            continue
        normalized = str(key).replace("\\", "/")
        base = os.path.basename(normalized)
        stem = os.path.splitext(base)[0].lower()
        replacements[normalized] = str(value)
        replacements[base] = str(value)
        replacements[stem] = str(value)
        replacements[normalized.lower()] = str(value)
        replacements[base.lower()] = str(value)

    tmp_path = f"{output_path}.tmp"
    with zipfile.ZipFile(source_epub_path, "r") as zin, zipfile.ZipFile(tmp_path, "w") as zout:
        infos = zin.infolist()
        replacement_by_html = {
            info.filename: _lookup_replacement(info.filename, replacements)
            for info in infos
            if info.filename.lower().endswith(HTML_EXTS)
        }
        preserved_image_refs = set()
        for html_name, replacement in replacement_by_html.items():
            if replacement is not None:
                preserved_image_refs.update(_replacement_image_refs(html_name, replacement, reverse_rename_map))
            elif _is_preserved_original_html(html_name):
                try:
                    preserved_image_refs.update(_html_image_refs(html_name, zin.read(html_name)))
                except Exception:
                    pass

        mimetype_info = next((info for info in infos if info.filename == "mimetype"), None)
        if mimetype_info is not None:
            mt_info = _copy_zip_info(mimetype_info)
            mt_info.compress_type = zipfile.ZIP_STORED
            zout.writestr(mt_info, zin.read(mimetype_info.filename))

        for info in infos:
            if info.filename == "mimetype":
                continue
            out_info = _copy_zip_info(info)
            lower_name = info.filename.lower()
            if lower_name.endswith(HTML_EXTS):
                replacement = replacement_by_html.get(info.filename)
                original_data = zin.read(info.filename)
                if replacement is None and _is_preserved_original_html(info.filename):
                    zout.writestr(out_info, original_data)
                else:
                    data = _html_with_ocr_body(
                        original_data,
                        replacement or "",
                        os.path.splitext(os.path.basename(info.filename))[0],
                        reverse_rename_map,
                    )
                    out_info.compress_type = zipfile.ZIP_DEFLATED
                    zout.writestr(out_info, data)
            elif lower_name.endswith(IMAGE_EXTS) and _zip_norm(info.filename) not in preserved_image_refs:
                continue
            else:
                zout.writestr(out_info, zin.read(info.filename))

    os.replace(tmp_path, output_path)
    return output_path


def _html_to_text(data: bytes) -> str:
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        text = str(data)
    try:
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "nav", "head", "title"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
    except Exception:
        text = re.sub(r"<[^>]+>", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", html.unescape(text or "")).strip()


def load_ocr_epub_text(ocr_epub_path: str, chapter_ref=None) -> str:
    """Load visible text from the OCR EPUB, optionally scoped to one chapter."""
    if not ocr_epub_path or not os.path.exists(ocr_epub_path):
        return ""

    chapter_file = ""
    if isinstance(chapter_ref, dict):
        chapter_file = str(chapter_ref.get("chapter_file") or "")
    elif chapter_ref:
        chapter_file = str(chapter_ref)
    target_base = os.path.basename(chapter_file).lower()
    target_stem = os.path.splitext(target_base)[0]

    parts = []
    with zipfile.ZipFile(ocr_epub_path, "r") as zf:
        html_infos = [info for info in zf.infolist() if info.filename.lower().endswith(HTML_EXTS)]
        if target_base or target_stem:
            for info in html_infos:
                base = os.path.basename(info.filename).lower()
                stem = os.path.splitext(base)[0]
                if base == target_base or stem == target_stem:
                    return _html_to_text(zf.read(info.filename))
        for info in html_infos:
            text = _html_to_text(zf.read(info.filename))
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip()
