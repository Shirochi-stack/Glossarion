import html
import os
import re
import zipfile
from typing import Dict, Optional

from bs4 import BeautifulSoup


HTML_EXTS = (".html", ".xhtml", ".htm")


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


def write_ocr_epub(source_epub_path: str, chapter_text_by_filename: Dict[str, str], output_path: Optional[str] = None) -> str:
    """Copy an EPUB and replace chapter HTML bodies with raw OCR/text content."""
    if not source_epub_path or not os.path.exists(source_epub_path):
        raise FileNotFoundError(source_epub_path)
    output_path = output_path or ocr_epub_path_for(source_epub_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    replacements = {}
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
        mimetype_info = next((info for info in infos if info.filename == "mimetype"), None)
        if mimetype_info is not None:
            mt_info = _copy_zip_info(mimetype_info)
            mt_info.compress_type = zipfile.ZIP_STORED
            zout.writestr(mt_info, zin.read(mimetype_info.filename))

        for info in infos:
            if info.filename == "mimetype":
                continue
            out_info = _copy_zip_info(info)
            replacement = _lookup_replacement(info.filename, replacements)
            if replacement is not None and info.filename.lower().endswith(HTML_EXTS):
                data = _text_to_xhtml(replacement, os.path.splitext(os.path.basename(info.filename))[0])
                out_info.compress_type = zipfile.ZIP_DEFLATED
                zout.writestr(out_info, data)
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
