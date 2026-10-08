"""Plain-text books in the Reader (UI_SPEC §3.11 "Modes": text and TXT workspace), no Flet.

The desktop has no in-app TXT reader (its Library hands a ``.txt`` to the system editor:
``library_core.plan_open_reader`` returns ``{"mode": "system"}``), so there is no desktop
reader code to move here. What this module reuses instead of copying:

* the TXT translation split of ``txt_processor.TextFileProcessor``: the section separator its
  ``create_output_structure`` writes between the translated sections of ``<stem>_translated.txt``,
  and the persisted split (``<workspace>/.cache/split.cache`` + ``word_count/``), restored with
  its own ``_load_split_cache`` (which also gives each section the pipeline's ``content_hash``);
* the pipeline's progress key (``translation_progress.json`` entries carry that ``content_hash``)
  and, as a fallback, its output names (``TransateKRtoEN.FileUtilities.create_chapter_filename``)
  to pair a raw section with its translation (pairing by number is wrong: chunk 11 of chapter 1
  is numbered 2.0);
* ``workspace_reader.build_workspace_reader_manifest`` for a workspace manifest's base fields and
  ``reader_doc``'s workspace placeholder.

Chapters follow the ``reader_doc.load_epub_chapters`` / ``load_workspace_chapters`` contracts, so
``ReaderSession`` pages, searches, bookmarks and pairs them (Bilingual) like any other book. A
standalone ``.txt`` is cut into paragraph-bounded sections of about ``READER_TXT_SECTION_CHARS``
characters (deterministic, unlike the token- and model-dependent ``ChapterSplitter``, so saved
positions stay valid); compiled translation output splits on its separator, one section per
translated section.

GUI-free; the backend modules are imported inside the functions only. Python 3.10 compatible.
"""

from __future__ import annotations

import codecs
import html as html_lib
import os
import re
from typing import Any, Callable, Mapping, Optional

__all__ = [
    "COMPILED_SECTION_SEPARATOR",
    "READER_SECTION_NAME",
    "READER_TXT_SECTION_CHARS",
    "UNTRANSLATED_TEXT",
    "build_text_workspace_manifest",
    "decode_text_bytes",
    "has_text_workspace",
    "is_compiled_text_output",
    "load_text_chapters",
    "load_text_workspace_chapters",
    "normalize_text",
    "section_filename",
    "section_title",
    "split_cache_path",
    "text_section_count",
    "text_to_reader_html",
    "workspace_section_count",
]

#: A standalone TXT is read in sections of about this many characters, cut at paragraph ends.
READER_TXT_SECTION_CHARS = 20000
#: ``txt_processor.TextFileProcessor.create_output_structure`` joins translated text sections with this.
COMPILED_SECTION_SEPARATOR = "\n\n" + "=" * 50 + "\n\n"
#: The name ``create_output_structure`` gives a compiled text translation (``<stem>_translated.txt``).
COMPILED_SUFFIX = "_translated.txt"
#: The Reader's own section names (``section_0001.txt``...); the translation's are ``section_1_0.txt``...
READER_SECTION_NAME = re.compile(r"^section_\d{4,}\.txt$", re.IGNORECASE)
UNTRANSLATED_TEXT = "This section has not been translated yet."

_SPLIT_CACHE = "split.cache"  # TextFileProcessor's translation split (never split_glossary.cache)
_PROGRESS = "translation_progress.json"
#: Byte order marks, longest first (the UTF-32 LE mark starts with the UTF-16 LE one).
_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)
#: Without a BOM and not UTF-8: the CJK code pages. Never BOM-less UTF-16: it decodes almost any
#: even-length bytes, so a CP949 / GBK file would come out as mojibake instead of reaching its codec.
#: "The first that decodes" cannot pick among these (GB18030 accepts almost any double-byte text and
#: CP949 many Shift-JIS pairs), so chardet / charset_normalizer (both pinned in the app) choose; this
#: order is only the fallback when neither names a codec that reads the text.
_CJK_CODECS = ("cp949", "gb18030", "shift_jis", "cp932", "big5", "euc_jp")
#: A detector's codec name (``codecs.lookup(name).name``) -> the codecs that read it, superset first
#: (CP932 has the NEC / IBM characters such as ① that Shift-JIS lacks; GB18030 contains GB2312 / GBK).
_CODEC_FAMILY = {
    "shift_jis": ("cp932", "shift_jis"),
    "cp932": ("cp932", "shift_jis"),
    "euc_jp": ("euc_jp",),
    "euc_kr": ("cp949", "euc_kr"),
    "cp949": ("cp949",),
    "gb2312": ("gb18030",),
    "gbk": ("gb18030",),
    "gb18030": ("gb18030",),
    "big5": ("big5", "cp950", "big5hkscs"),
    "cp950": ("cp950", "big5", "big5hkscs"),
    "big5hkscs": ("big5hkscs",),
}
#: The detectors read this much of the file (chardet is pure Python: ~40-80 ms per 32 KB).
_DETECT_SAMPLE = 32 * 1024
_C1_CONTROLS = re.compile("[\u0080-\u009f]")
_BLANK_LINE = re.compile(r"\n[^\S\n]*\n")
_BLANK_RUN = re.compile(r"\n(?:[^\S\n]*\n)+")
#: Where an over-long paragraph may be cut, best first (only paragraphs longer than a section).
_CUT_POINTS = ("\n", "　", " ", "\t", "。", ". ", "!", "?", "！", "？")


# ---------------------------------------------------------------------------
# Text -> reader HTML
# ---------------------------------------------------------------------------


def _decodes(data: bytes, codec: str, final: bool) -> bool:
    """Whether ``data`` reads as text in ``codec``: it decodes strictly (``final`` False: a sample
    whose last character may be cut off) to no C1 control character (CP932 reads a stray byte
    0x80 as U+0080, so Latin-1 bytes "decode"; no real text has C1 controls)."""
    try:
        text = codecs.getincrementaldecoder(codec)().decode(data, final)
    except (UnicodeDecodeError, LookupError):
        return False
    return _C1_CONTROLS.search(text) is None


def _detected_codecs(sample: bytes, candidates: list) -> list:
    """The codecs chardet, then charset_normalizer (limited to ``candidates``), name for
    ``sample``, each expanded to its family; [] when neither is importable or sure."""
    names: list = []
    try:
        import chardet  # pinned in the app; the detector for short texts

        names.append(chardet.detect(sample).get("encoding") or "")
    except Exception:
        pass
    try:
        import charset_normalizer  # pinned in the app

        best = charset_normalizer.from_bytes(sample, cp_isolation=list(candidates)).best()
        if best is not None:
            names.append(best.encoding or "")
    except Exception:
        pass
    found: list = []
    for name in names:
        try:
            key = codecs.lookup(name).name if name else ""
        except LookupError:
            continue
        found.extend(_CODEC_FAMILY.get(key, ()))
    return list(dict.fromkeys(found))


def decode_text_bytes(data: bytes) -> str:
    """A text file's bytes as str: BOM (UTF-8 / UTF-16 / UTF-32), else strict UTF-8, else the CJK
    code page (CP949, GB18030 / GBK, Shift-JIS / CP932, Big5, EUC-JP) the detectors pick among the
    ones that read the text, else UTF-8 with replacement characters."""
    data = bytes(data or b"")
    for bom, codec in _BOMS:
        if data.startswith(bom):
            return data[len(bom):].decode(codec, errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    sample = data[:_DETECT_SAMPLE]
    final = len(data) <= _DETECT_SAMPLE
    candidates = [codec for codec in _CJK_CODECS if _decodes(sample, codec, final)]
    if not candidates:
        return data.decode("utf-8", errors="replace")
    detected = [codec for codec in _detected_codecs(sample, candidates) if _decodes(sample, codec, final)]
    for codec in detected or candidates:
        try:
            return data.decode(codec)
        except UnicodeDecodeError:
            continue
    if detected:  # a stray byte past the sample: the detected code page with a replacement there
        return data.decode(detected[0], errors="replace")
    return data.decode("utf-8", errors="replace")


def normalize_text(text: Any) -> str:
    """LF line ends, no leading BOM character, no NUL characters."""
    value = str(text or "")
    if value.startswith("﻿"):
        value = value[1:]
    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def _units(text: str) -> tuple:
    """``(blocks, units)``: blank-line separated blocks when the text has blank lines (``blocks``
    True; a block's single newlines are line breaks), else one unit per non-empty line."""
    if _BLANK_LINE.search(text):
        return True, [unit.strip("\n") for unit in _BLANK_RUN.split(text) if unit.strip()]
    return False, [line for line in text.split("\n") if line.strip()]


def _unit_html(unit: str, blocks: bool) -> str:
    lines = unit.split("\n") if blocks else [unit]
    # quote=False: text needs no quote escapes, and the sanitiser's re-serialisation (when a text
    # merely mentions "onclick=" / "javascript:") then gives the same markup back
    return "<p>" + "<br/>".join(html_lib.escape(line, quote=False) for line in lines) + "</p>"


def _units_html(units: list, blocks: bool) -> str:
    return "\n".join(_unit_html(unit, blocks) for unit in units)


def text_to_reader_html(text: Any, title: str = "") -> str:
    """Plain text as reader HTML: blank-line separated blocks become ``<p>`` (their single
    newlines ``<br/>``); text without blank lines gets one ``<p>`` per non-empty line. Everything is
    escaped, so markup in a ``.txt`` shows as text. ``title`` adds an ``<h2>`` above it."""
    blocks, units = _units(normalize_text(text))
    body = _units_html(units, blocks)
    if title:
        body = f"<h2>{html_lib.escape(str(title), quote=False)}</h2>\n" + body
    return body


# ---------------------------------------------------------------------------
# Standalone TXT sections
# ---------------------------------------------------------------------------


def section_filename(number: int) -> str:
    """The Reader's stable name for section ``number`` (1-based): reading positions use it."""
    return f"section_{int(number):04d}.txt"


def section_title(number: int) -> str:
    return f"Section {int(number)}"


def is_compiled_text_output(path: str) -> bool:
    """A compiled translation (``<stem>_translated.txt``, or a ``.txt`` next to a translation's
    ``translation_progress.json``): its sections are the translation's, cut on the separator."""
    name = os.path.basename(str(path or "")).lower()
    folder = os.path.dirname(os.path.abspath(str(path or "")))
    return name.endswith(COMPILED_SUFFIX) or os.path.isfile(os.path.join(folder, _PROGRESS))


def _cut_long(unit: str, limit: int) -> list:
    """A paragraph longer than a section, cut near ``limit`` at a line end / space / sentence end
    (files without paragraph breaks would otherwise be one huge page)."""
    if len(unit) <= limit:
        return [unit]
    pieces = []
    rest = unit
    while len(rest) > limit:
        window = rest[:limit]
        cut = -1
        for mark in _CUT_POINTS:
            found = window.rfind(mark)
            if found >= limit // 2:
                cut = found + len(mark)
                break
        if cut <= 0:
            cut = limit
        head, rest = rest[:cut].rstrip(), rest[cut:].lstrip()
        if head.strip():
            pieces.append(head)
    if rest.strip():
        pieces.append(rest)
    return pieces


def _pack(units: list, limit: int) -> list:
    """Paragraph units packed into sections of at most ``limit`` characters (cut only between
    paragraphs, except a paragraph longer than a section)."""
    sections: list = []
    current: list = []
    size = 0
    for unit in units:
        for piece in _cut_long(unit, limit):
            if current and size + len(piece) > limit:
                sections.append(current)
                current, size = [], 0
            current.append(piece)
            size += len(piece)
    if current:
        sections.append(current)
    return sections or [[]]


def _split_sections(text: str, compiled: bool) -> list:
    """``[(blocks, units)]`` per section of normalised ``text``."""
    if compiled and COMPILED_SECTION_SEPARATOR in text:
        return [_units(part) for part in text.split(COMPILED_SECTION_SEPARATOR)]
    blocks, units = _units(text)
    return [(blocks, section) for section in _pack(units, READER_TXT_SECTION_CHARS)]


def _read_text(path: str) -> str:
    try:
        with open(path, "rb") as stream:
            return decode_text_bytes(stream.read())
    except OSError:
        return ""


def load_text_chapters(path: str, *, should_stop: Optional[Callable[[], bool]] = None) -> Optional[tuple]:
    """``(chapters [(title, html)], {}, filenames)`` for a ``.txt`` (the ``load_epub_chapters``
    contract); None when ``should_stop`` fired. Raises ``OSError`` when the file cannot be read.

    Compiled translation output splits on its section separator; any other text is cut into
    paragraph-bounded sections of about ``READER_TXT_SECTION_CHARS`` characters. Sections are
    titled "Section N" and named ``section_0001.txt``...; an empty file is one empty section."""
    with open(path, "rb") as stream:
        text = normalize_text(decode_text_bytes(stream.read()))
    chapters: list = []
    filenames: list = []
    for number, (blocks, units) in enumerate(_split_sections(text, is_compiled_text_output(path)), 1):
        if should_stop is not None and should_stop():
            return None
        chapters.append((section_title(number), _units_html(units, blocks)))
        filenames.append(section_filename(number))
    return chapters, {}, filenames


def text_section_count(path: str) -> int:
    """How many sections ``load_text_chapters`` gives ``path`` (0 when it cannot be read)."""
    try:
        with open(path, "rb") as stream:
            text = normalize_text(decode_text_bytes(stream.read()))
    except OSError:
        return 0
    return len(_split_sections(text, is_compiled_text_output(path)))


# ---------------------------------------------------------------------------
# TXT translation workspaces
# ---------------------------------------------------------------------------


def split_cache_path(workspace: str) -> str:
    """``<workspace>/.cache/split.cache`` (or the pre-``.cache`` location), "" when absent."""
    for candidate in (os.path.join(workspace, ".cache", _SPLIT_CACHE), os.path.join(workspace, _SPLIT_CACHE)):
        if os.path.isfile(candidate):
            return candidate
    return ""


def _split_cache_meta(workspace: str) -> tuple:
    """``(cache path, source_hash, chapter metadata)`` of the translation split, or ``("", "", [])``."""
    path = split_cache_path(str(workspace or ""))
    if not path:
        return "", "", []
    from pathlib import Path

    from workspace_reader import _load_json

    cache = _load_json(Path(path), {}) or {}
    if not isinstance(cache, Mapping):
        return "", "", []
    chapters = cache.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        return "", "", []
    return path, str(cache.get("source_hash") or ""), [dict(c) for c in chapters if isinstance(c, Mapping)]


def has_text_workspace(workspace: str) -> bool:
    """Whether ``workspace`` holds a translation split the Reader can show (split.cache with every
    ``word_count`` section file present). Reads metadata only."""
    _path, _hash, chapters = _split_cache_meta(workspace)
    if not chapters:
        return False
    word_count = os.path.join(str(workspace), "word_count")
    names = [os.path.basename(str(c.get("filename") or "")) for c in chapters]
    return all(name and os.path.isfile(os.path.join(word_count, name)) for name in names)


def workspace_section_count(workspace: str) -> int:
    """Sections in a TXT workspace's translation split (0 without one)."""
    return len(_split_cache_meta(workspace)[2])


def _split_sections_restored(workspace: str) -> list:
    """The ordered split sections with ``body`` and ``content_hash`` (``_load_split_cache``)."""
    path, source_hash, chapters = _split_cache_meta(workspace)
    if not chapters:
        return []
    from txt_processor import TextFileProcessor

    # __new__ skips __init__, which builds a tiktoken ChapterSplitter the restore never uses.
    processor = TextFileProcessor.__new__(TextFileProcessor)
    restored = processor._load_split_cache(path, source_hash, os.path.join(workspace, "word_count"))
    return [dict(section) for section in (restored or []) if isinstance(section, Mapping)]


def _output_path(workspace: str, output_file: Any) -> str:
    """An existing translated file of a progress entry ("" otherwise)."""
    name = str(output_file or "").strip()
    if not name:
        return ""
    candidate = os.path.normpath(name if os.path.isabs(name) else os.path.join(workspace, name))
    return candidate if os.path.isfile(candidate) else ""


def _pair_by_hash(workspace: str, candidates: list) -> tuple:
    """``(translated path, status)`` from the progress entries sharing a section's content hash:
    a completed entry with its file first, then any entry with its file, else the first status."""
    found = [(_output_path(workspace, entry.get("output_file")), str(entry.get("status") or ""))
             for entry in candidates]
    with_file = [pair for pair in found if pair[0]]
    for path, status in with_file:
        if status.strip().lower() == "completed":
            return path, status
    if with_file:
        return with_file[0]
    return ("", found[0][1]) if found else ("", "")


def _name_variants(name: str) -> list:
    """An output name with and without the ``response_`` prefix (``RETAIN_SOURCE_EXTENSION`` drops it)."""
    base = os.path.basename(str(name or ""))
    if not base:
        return []
    other = base[len("response_"):] if base.startswith("response_") else "response_" + base
    return [base, other]


def _pair_by_output_name(workspace: str, sections: list, entries: list, by_output: Mapping) -> None:
    """Sections without a hash match: the pipeline's own output name, when such a file exists.

    ``TransateKRtoEN`` is large, so it is imported only when the workspace holds a text file
    that could still be one of these sections' translations."""
    try:
        on_disk = {name.casefold(): name for name in os.listdir(workspace) if name.lower().endswith(".txt")}
    except OSError:
        return
    paired = {os.path.basename(e["translated_path"]).casefold() for e in entries if e.get("translated_path")}
    candidates = {key for key in on_disk if key not in paired and key != "source_epub.txt"
                  and not key.endswith(COMPILED_SUFFIX)}
    if not candidates:
        return
    try:
        from TransateKRtoEN import FileUtilities
    except Exception:
        return
    for index, entry in enumerate(entries):
        if entry.get("translated_path"):
            continue
        meta = {k: v for k, v in sections[index].items() if k not in ("body", "content_hash")}
        try:
            name = FileUtilities.create_chapter_filename(meta)
        except Exception:
            continue
        for variant in _name_variants(name):
            key = variant.casefold()
            if key in candidates:
                entry["translated_path"] = os.path.join(workspace, on_disk[key])
                entry["status"] = str((by_output.get(key) or {}).get("status") or entry.get("status") or "")
                candidates.discard(key)
                break


def build_text_workspace_manifest(workspace: str, *, source_path: Optional[str] = None,
                                  base: Optional[Mapping[str, Any]] = None) -> dict:
    """The reader manifest of a TXT translation workspace (``source_format`` "txt").

    The base fields come from ``workspace_reader.build_workspace_reader_manifest`` (or ``base``,
    an already built one). ``entries`` are the translation split's sections in order (none before
    the first translation run splits the book): ``raw_path`` (``word_count/<filename>``) and
    ``translated_path``, the output of the progress entry with the section's ``content_hash`` (the
    pipeline's key), else the pipeline's output name; a path is set only when the file exists."""
    if base is None:
        from workspace_reader import build_workspace_reader_manifest

        base = build_workspace_reader_manifest(workspace, source_path=source_path or None)
    manifest = dict(base)
    root = str(manifest.get("workspace") or os.path.abspath(str(workspace)))
    manifest["source_format"] = "txt"
    sections = _split_sections_restored(root)
    entries: list = []
    if sections:
        from pathlib import Path

        from workspace_reader import _load_json

        progress = _load_json(Path(root) / _PROGRESS, {}) or {}
        chapters = progress.get("chapters") if isinstance(progress, Mapping) else {}
        by_hash: dict = {}
        by_output: dict = {}
        for entry in (chapters if isinstance(chapters, Mapping) else {}).values():
            if not isinstance(entry, Mapping):
                continue
            if entry.get("content_hash"):
                by_hash.setdefault(str(entry["content_hash"]), []).append(entry)
            if entry.get("output_file"):
                by_output[os.path.basename(str(entry["output_file"])).casefold()] = entry
        for number, section in enumerate(sections, 1):
            filename = os.path.basename(str(section.get("filename") or ""))
            translated, status = _pair_by_hash(root, by_hash.get(str(section.get("content_hash") or ""), []))
            entries.append({
                "key": filename,
                "filename": filename,
                "original_filename": filename,
                "title": section_title(number),
                "raw_path": os.path.join(root, "word_count", filename),
                "translated_path": translated,
                "status": status,
                "content_hash": str(section.get("content_hash") or ""),
                "pdf_toc_section": False,
                "pdf_section_id": "",
                "pdf_start_page": None,
                "pdf_end_page": None,
            })
        if any(not e["translated_path"] for e in entries):
            _pair_by_output_name(root, sections, entries, by_output)
    manifest["entries"] = entries
    return manifest


def _untranslated_html(title: str, raw_html: str) -> str:
    """The workspace placeholder (``reader_doc``) with the section's raw text below it."""
    from reader_doc import _workspace_reader_placeholder

    page = _workspace_reader_placeholder(title, UNTRANSLATED_TEXT)
    return page.replace("</body>", raw_html + "</body>", 1) if raw_html else page


def load_text_workspace_chapters(manifest: Mapping[str, Any], *,
                                 should_stop: Optional[Callable[[], bool]] = None) -> Optional[tuple]:
    """``(raw_chapters, translated_chapters, filenames)`` of a TXT workspace manifest (the
    ``reader_doc.load_workspace_chapters`` contract); None when ``should_stop`` fired. Both sides go
    through ``text_to_reader_html``; an untranslated section shows the placeholder above its raw text."""
    raw_chapters: list = []
    translated_chapters: list = []
    filenames: list = []
    for number, entry in enumerate(manifest.get("entries") or [], 1):
        if should_stop is not None and should_stop():
            return None
        title = str(entry.get("title") or section_title(number))
        raw_html = text_to_reader_html(_read_text(str(entry.get("raw_path") or "")))
        translated_path = str(entry.get("translated_path") or "")
        if translated_path and os.path.isfile(translated_path):
            translated_html = text_to_reader_html(_read_text(translated_path))
        else:
            translated_html = _untranslated_html(title, raw_html)
        raw_chapters.append((title, raw_html))
        translated_chapters.append((title, translated_html))
        filenames.append(str(entry.get("filename") or ""))
    return raw_chapters, translated_chapters, filenames
