# -*- coding: utf-8 -*-
"""Glossary usage helpers for source-chapter footnotes and editor filtering."""

import csv
import html
import json
import os
import posixpath
import re
import time
import unicodedata
import zipfile
from io import StringIO
from urllib.parse import unquote
from xml.etree import ElementTree as ET

try:
    from glossary_compressor import _text_contains_term
except Exception:  # pragma: no cover - fallback for isolated imports
    def _text_contains_term(text, term, is_character=False):
        if not text or not term:
            return False
        if term in text:
            return True
        min_token_len = 1 if is_character else 2
        if " " in term:
            return any(len(tok) >= min_token_len and tok in text for tok in term.split())
        return False


CHECK_PREFIX = "\u2705"
WARNING_PREFIX = "\u26a0\ufe0f"
GLOSSARY_SEP = "\x1F"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _norm_text(value):
    return str(value or "").strip()


def _entry_identity(entry):
    raw = _norm_text(entry.get("raw_name") or entry.get("original_name") or entry.get("original"))
    translated = _norm_text(
        entry.get("translated_name")
        or entry.get("translation")
        or entry.get("translated")
        or entry.get("name")
    )
    gender = _norm_text(entry.get("gender"))
    typ = _norm_text(entry.get("type") or "terms")
    return (raw.casefold(), translated.casefold(), gender.casefold(), typ.casefold())


def _normalize_entry(entry, source_index, source_key=None):
    if not isinstance(entry, dict):
        entry = {"raw_name": str(entry)}
    raw_name = _norm_text(entry.get("raw_name") or entry.get("original_name") or entry.get("original") or source_key)
    translated_name = _norm_text(
        entry.get("translated_name")
        or entry.get("translation")
        or entry.get("translated")
        or entry.get("name")
    )
    entry_type = _norm_text(entry.get("type") or "terms")
    gender = _norm_text(entry.get("gender"))
    description = _norm_text(entry.get("description") or entry.get("notes") or entry.get("context"))
    normalized = dict(entry)
    normalized.update(
        {
            "source_index": int(source_index),
            "raw_name": raw_name,
            "translated_name": translated_name,
            "type": entry_type,
            "gender": gender,
            "description": description,
            "identity": _entry_identity(
                {
                    "raw_name": raw_name,
                    "translated_name": translated_name,
                    "gender": gender,
                    "type": entry_type,
                }
            ),
        }
    )
    return normalized


def _split_head_desc(text):
    paren_depth = 0
    bracket_depth = 0
    for idx, ch in enumerate(text):
        if ch == "(" and bracket_depth == 0:
            paren_depth += 1
        elif ch == ")" and bracket_depth == 0 and paren_depth > 0:
            paren_depth -= 1
        elif ch == "[" and paren_depth == 0:
            bracket_depth += 1
        elif ch == "]" and paren_depth == 0 and bracket_depth > 0:
            bracket_depth -= 1
        elif ch == ":" and paren_depth == 0 and bracket_depth == 0:
            return text[:idx].rstrip(), text[idx + 1 :].strip()
    return text, ""


def _strip_custom_tails(text):
    while True:
        match = re.search(r"\s+\(([^()]*)\)\s*$", text)
        if not match or ":" not in match.group(1):
            return text.rstrip()
        text = text[: match.start()].rstrip()


def _parse_token_entry_line(line, current_type="terms", source_index=0):
    body = line[2:].strip() if line.lstrip().startswith("* ") else line.strip()
    head, desc = _split_head_desc(body)
    head = _strip_custom_tails(head)
    gender = ""
    gender_match = re.search(r"\s*\[([^\]]*)\]\s*$", head)
    if gender_match:
        gender = _norm_text(gender_match.group(1))
        head = _strip_custom_tails(head[: gender_match.start()].rstrip())

    raw_name = ""
    translated_name = ""
    equal_match = re.match(r"^(?P<raw>.+?)\s*=\s*(?P<translated>.+?)\s*$", head)
    if equal_match:
        raw_name = _norm_text(equal_match.group("raw"))
        translated_name = _norm_text(equal_match.group("translated"))
    else:
        legacy_match = re.match(r"^(?P<translated>.*)\s+\((?P<raw>.*?)\)\s*$", head)
        if legacy_match:
            raw_name = _norm_text(legacy_match.group("raw"))
            translated_name = _norm_text(legacy_match.group("translated"))

    if not raw_name and not translated_name:
        return None

    return _normalize_entry(
        {
            "type": current_type,
            "raw_name": raw_name,
            "translated_name": translated_name,
            "gender": gender,
            "description": desc,
        },
        source_index,
    )


def _section_to_type(section):
    value = _norm_text(section).strip("=").strip().lower()
    if not value:
        return "terms"
    singular = {
        "characters": "character",
        "character": "character",
        "terms": "terms",
        "term": "terms",
        "titles": "titles",
        "title": "titles",
        "locations": "locations",
        "location": "locations",
        "organizations": "organizations",
        "organization": "organizations",
        "items": "items",
        "item": "items",
        "abilities": "abilities",
        "abilitys": "abilities",
        "ability": "abilities",
        "nicknames": "nicknames",
        "nickname": "nicknames",
        "surnames": "surnames",
        "surname": "surnames",
    }
    return singular.get(value, value)


def parse_glossary_content(content, suffix=".csv"):
    """Parse glossary content into normalized entry dictionaries."""
    suffix = str(suffix or "").lower()
    if suffix == ".json":
        data = json.loads(content) if isinstance(content, str) else content
        return parse_json_glossary(data)
    return parse_csv_glossary(content)


def parse_glossary_file(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".json":
        return parse_json_glossary(json.loads(content))
    return parse_csv_glossary(content)


def parse_json_glossary(data):
    entries = []
    if isinstance(data, dict):
        source = data.get("entries", data)
        if isinstance(source, dict):
            for idx, (key, value) in enumerate(source.items()):
                if isinstance(value, dict):
                    entry = dict(value)
                    entry.setdefault("raw_name", key)
                else:
                    entry = {"raw_name": key, "translated_name": value}
                entries.append(_normalize_entry(entry, idx, source_key=key))
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            if isinstance(value, dict):
                entries.append(_normalize_entry(value, idx))
    return [entry for entry in entries if entry.get("raw_name")]


def parse_csv_glossary(content):
    lines = str(content or "").splitlines()
    token_style = any(line.lstrip().startswith("===") or line.lstrip().startswith("* ") for line in lines)
    token_style = token_style or bool(lines and lines[0].lower().startswith("glossary columns:"))
    if token_style:
        return parse_token_csv_glossary(lines)

    sep = GLOSSARY_SEP if GLOSSARY_SEP in str(content or "") else ","
    if sep == GLOSSARY_SEP:
        rows = [[part.strip() for part in line.split(sep)] for line in lines if line.strip()]
    else:
        rows = list(csv.reader(StringIO(str(content or ""))))

    entries = []
    header = []
    data_start = 0
    if rows:
        first = [str(cell or "").strip().lower() for cell in rows[0]]
        if "raw_name" in first or (first and first[0] == "type"):
            header = first
            data_start = 1

    def _cell(row, name, fallback_idx=None):
        if name in header:
            idx = header.index(name)
            return row[idx] if len(row) > idx else ""
        if fallback_idx is not None and len(row) > fallback_idx:
            return row[fallback_idx]
        return ""

    for idx, row in enumerate(rows[data_start:]):
        if not row or len(row) < 2:
            continue
        entry = {
            "type": _cell(row, "type", 0) or "terms",
            "raw_name": _cell(row, "raw_name", 1 if len(row) >= 3 else 0),
            "translated_name": _cell(row, "translated_name", 2 if len(row) >= 3 else 1),
            "gender": _cell(row, "gender", 3),
            "description": _cell(row, "description", 4),
        }
        if header:
            for col_idx, name in enumerate(header):
                if name and col_idx < len(row) and name not in entry:
                    entry[name] = row[col_idx]
        normalized = _normalize_entry(entry, idx)
        if normalized.get("raw_name"):
            entries.append(normalized)
    return entries


def parse_token_csv_glossary(lines):
    entries = []
    current_type = "terms"
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("glossary columns:"):
            continue
        if stripped.startswith("===") and stripped.endswith("==="):
            current_type = _section_to_type(stripped)
            continue
        if not stripped.startswith("* "):
            continue
        entry = _parse_token_entry_line(stripped, current_type=current_type, source_index=len(entries))
        if entry and entry.get("raw_name"):
            entries.append(entry)
    return entries


def html_to_text(markup):
    text = str(markup or "")
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text("\n")
    except Exception:
        text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _read_epub_opf(zf):
    try:
        container = ET.fromstring(zf.read("META-INF/container.xml"))
        ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
        rootfile = container.find(".//c:rootfile", ns)
        if rootfile is not None and rootfile.get("full-path"):
            return rootfile.get("full-path")
    except Exception:
        pass
    return next((name for name in zf.namelist() if name.lower().endswith(".opf")), None)


def _epub_member_path(opf_path, href):
    opf_dir = posixpath.dirname(opf_path or "")
    href = unquote(str(href or ""))
    return posixpath.normpath(posixpath.join(opf_dir, href)).lstrip("/")


def _special_file_sets():
    kw_env = os.environ.get("SPECIAL_FILE_KEYWORDS", "")
    keywords = [k.strip().lower() for k in kw_env.split(",") if k.strip()] if kw_env else [
        "title",
        "toc",
        "copyright",
        "preface",
        "nav",
        "message",
        "notice",
        "colophon",
        "dedication",
        "epigraph",
        "foreword",
        "acknowledgment",
        "author",
        "appendix",
        "bibliography",
    ]
    exact_env = os.environ.get("SPECIAL_FILE_EXACT", "")
    exact = [k.strip().lower() for k in exact_env.split(",") if k.strip()] if exact_env else [
        "index",
        "glossary",
        "glossary_extension",
    ]
    return keywords, exact


def _is_special_basename(basename, keywords=None, exact=None):
    keywords = keywords if keywords is not None else _special_file_sets()[0]
    exact = exact if exact is not None else _special_file_sets()[1]
    name_noext = os.path.splitext(os.path.basename(str(basename or "")))[0]
    name_lower = name_noext.lower()
    name_stripped = re.sub(r"\d+$", "", name_lower).rstrip("_- ")
    if name_lower in exact:
        return True
    if any(kw in name_lower for kw in keywords):
        has_digits = bool(re.search(r"\d", name_noext))
        if not has_digits or any(kw == name_stripped or kw in name_stripped for kw in keywords):
            return True
    return False


def read_epub_spine_chapters(epub_path, translate_special=False, include_text=True):
    """Return source chapters using the same special-file filtering as Progress Manager."""
    chapters = []
    if not epub_path or not os.path.exists(epub_path) or not str(epub_path).lower().endswith(".epub"):
        return chapters
    keywords, exact = _special_file_sets()
    with zipfile.ZipFile(epub_path, "r") as zf:
        opf_path = _read_epub_opf(zf)
        if not opf_path:
            return chapters
        opf_xml = ET.fromstring(zf.read(opf_path))
        ns = {"opf": "http://www.idpf.org/2007/opf"}
        id_to_item = {}
        html_types = {"application/xhtml+xml", "text/html", "application/html+xml"}
        for item in opf_xml.findall(".//opf:manifest/opf:item", ns):
            media_type = item.get("media-type", "")
            href = item.get("href", "")
            item_id = item.get("id", "")
            if item_id and href and media_type in html_types:
                id_to_item[item_id] = href
        for opf_pos, itemref in enumerate(opf_xml.findall(".//opf:spine/opf:itemref", ns), start=1):
            href = id_to_item.get(itemref.get("idref", ""))
            if not href:
                continue
            member_path = _epub_member_path(opf_path, href)
            basename = os.path.basename(member_path)
            if not translate_special and _is_special_basename(basename, keywords, exact):
                continue
            text = ""
            if include_text:
                try:
                    raw = zf.read(member_path)
                    markup = raw.decode("utf-8", errors="ignore")
                    text = html_to_text(markup)
                except Exception:
                    text = ""
            chapters.append(
                {
                    "chapter_index": len(chapters),
                    "spine_number": opf_pos,
                    "filename": basename,
                    "member_path": member_path,
                    "text": text,
                }
            )
    return chapters


def entry_matches_text(entry, source_text):
    raw_name = _norm_text(entry.get("raw_name"))
    if not raw_name:
        return False
    entry_type = _norm_text(entry.get("type")).lower()
    gender = _norm_text(entry.get("gender"))
    is_character = entry_type in {"character", "characters", "title", "titles", "nickname", "nicknames"} or bool(gender)
    return _text_contains_term(source_text, raw_name, is_character=is_character)


def match_entries_for_text(entries, source_text):
    return [entry for entry in entries if entry_matches_text(entry, source_text)]


def build_usage_index(entries, chapters):
    usage = {int(entry["source_index"]): set() for entry in entries}
    chapter_matches = {}
    for chapter in chapters:
        ci = int(chapter.get("chapter_index", len(chapter_matches)))
        matches = match_entries_for_text(entries, chapter.get("text", ""))
        chapter_matches[ci] = matches
        for entry in matches:
            usage.setdefault(int(entry["source_index"]), set()).add(ci)
    return usage, chapter_matches


def _fold_output_match_text(value):
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    value = "".join(ch if ch.isalnum() else " " for ch in value)
    return _WHITESPACE_RE.sub(" ", value).strip()


def _output_contains_term(output_text, term):
    term = _WHITESPACE_RE.sub(" ", _norm_text(term)).casefold()
    if not term:
        return False
    text = _WHITESPACE_RE.sub(" ", str(output_text or "")).casefold()
    if not text:
        return False
    if all(ord(ch) < 128 for ch in term) and (term[0].isalnum() or term[-1].isalnum()):
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])", text):
            return True
    elif term in text:
        return True
    folded_term = _fold_output_match_text(term)
    folded_text = _fold_output_match_text(text)
    if not folded_term or not folded_text:
        return False
    if all(ord(ch) < 128 for ch in folded_term):
        return re.search(r"(?<![A-Za-z0-9_])" + re.escape(folded_term) + r"(?![A-Za-z0-9_])", folded_text) is not None
    return folded_term in folded_text


def prepare_translated_output_text(output_text):
    """Precompute expensive normalized output-text forms for repeated entry matching."""
    text = _WHITESPACE_RE.sub(" ", str(output_text or "")).casefold()
    return {
        "text": text,
        "folded": _fold_output_match_text(text),
    }


def _output_contains_term_prepared(prepared_output, term):
    term = _WHITESPACE_RE.sub(" ", _norm_text(term)).casefold()
    if not term:
        return False
    text = prepared_output.get("text", "") if isinstance(prepared_output, dict) else ""
    if not text:
        return False
    if all(ord(ch) < 128 for ch in term) and (term[0].isalnum() or term[-1].isalnum()):
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])", text):
            return True
    elif term in text:
        return True
    folded_term = _fold_output_match_text(term)
    folded_text = prepared_output.get("folded", "") if isinstance(prepared_output, dict) else ""
    if not folded_term or not folded_text:
        return False
    if all(ord(ch) < 128 for ch in folded_term):
        return re.search(r"(?<![A-Za-z0-9_])" + re.escape(folded_term) + r"(?![A-Za-z0-9_])", folded_text) is not None
    return folded_term in folded_text


def entry_matches_prepared_translated_output(entry, prepared_outputs):
    candidates = [
        entry.get("translated_name"),
        entry.get("translated"),
        entry.get("name"),
    ]
    if not any(_norm_text(value) for value in candidates):
        candidates.extend(
            [
                entry.get("raw_name"),
                entry.get("original_name"),
                entry.get("original"),
            ]
        )
    return any(
        _output_contains_term_prepared(prepared_output, value)
        for prepared_output in prepared_outputs or []
        for value in candidates
    )


def entry_matches_translated_output(entry, output_text):
    candidates = [
        entry.get("translated_name"),
        entry.get("translated"),
        entry.get("name"),
    ]
    if not any(_norm_text(value) for value in candidates):
        candidates.extend(
            [
                entry.get("raw_name"),
                entry.get("original_name"),
                entry.get("original"),
            ]
        )
    return any(_output_contains_term(output_text, value) for value in candidates)


def build_translated_usage_index(entries, output_texts):
    usage = {int(entry["source_index"]): set() for entry in entries}
    for text_idx, output_text in enumerate(output_texts or []):
        for entry in entries:
            if entry_matches_translated_output(entry, output_text):
                usage.setdefault(int(entry["source_index"]), set()).add(text_idx)
    return usage


def read_translated_output_text(path):
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "rb") as f:
        raw_bytes = f.read()
    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = raw_bytes.decode("utf-8", errors="replace")
    if os.path.splitext(path)[1].lower() in {".html", ".htm", ".xhtml", ".xml"}:
        content = html_to_text(content)
    return content


def chapter_output_text(output_dir, chapter, progress_data=None):
    meta = _chapter_metadata(progress_data if isinstance(progress_data, dict) else {}, chapter)
    filename = meta.get("filename") or chapter.get("filename") or ""
    candidates = []
    if filename:
        if os.path.isabs(filename):
            candidates.append(filename)
        if output_dir:
            candidates.append(os.path.join(output_dir, filename))
            candidates.append(os.path.join(output_dir, os.path.basename(filename)))
            stem, ext = os.path.splitext(os.path.basename(filename))
            if stem and not ext:
                for suffix in (".xhtml", ".html", ".htm", ".txt", ".md"):
                    candidates.append(os.path.join(output_dir, stem + suffix))
    seen = set()
    for candidate in candidates:
        norm = os.path.normcase(os.path.abspath(candidate))
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isfile(candidate):
            return read_translated_output_text(candidate), candidate
    return None, candidates[0] if candidates else ""


def extracted_entry_match_keys(progress_data, chapter_index):
    """Return loose keys the AI extracted for a chapter from progress/tracker data."""
    keys = {"identities": set(), "raw_names": set(), "translated_names": set()}
    if not isinstance(progress_data, dict):
        return keys
    extracted = progress_data.get("chapter_extracted_entries") or progress_data.get("chapter_entry_index") or {}
    values = None
    if isinstance(extracted, dict):
        values = extracted.get(str(chapter_index))
        if values is None:
            values = extracted.get(chapter_index)
    if isinstance(values, list):
        for value in values:
            if isinstance(value, dict):
                normalized = _normalize_entry(value, 0)
                keys["identities"].add(normalized.get("identity"))
                raw = normalized.get("raw_name", "").casefold()
                translated = normalized.get("translated_name", "").casefold()
                if raw:
                    keys["raw_names"].add(raw)
                if translated:
                    keys["translated_names"].add(translated)
            elif isinstance(value, str):
                raw = value.strip().casefold()
                if raw:
                    keys["raw_names"].add(raw)
    keys["identities"].discard(None)
    return keys


def extracted_entry_identities(progress_data, chapter_index):
    """Return legacy exact identities for callers that still need them."""
    return extracted_entry_match_keys(progress_data, chapter_index).get("identities", set())


def _has_extracted_keys(extracted_keys):
    return any(extracted_keys.get(name) for name in ("identities", "raw_names", "translated_names"))


def entry_was_extracted(entry, extracted_keys):
    if not isinstance(entry, dict) or not isinstance(extracted_keys, dict):
        return False
    identity = entry.get("identity")
    raw = _norm_text(entry.get("raw_name")).casefold()
    translated = _norm_text(entry.get("translated_name")).casefold()
    return (
        bool(identity and identity in extracted_keys.get("identities", set()))
        or bool(raw and raw in extracted_keys.get("raw_names", set()))
        or bool(translated and translated in extracted_keys.get("translated_names", set()))
    )


def _format_entry_line(entry, skipped=False, confirmed=False):
    if skipped:
        prefix = f"- {WARNING_PREFIX} "
    elif confirmed:
        prefix = f"- {CHECK_PREFIX} "
    else:
        prefix = "- "
    raw = _norm_text(entry.get("raw_name"))
    translated = _norm_text(entry.get("translated_name"))
    typ = _norm_text(entry.get("type"))
    gender = _norm_text(entry.get("gender"))
    desc = _norm_text(entry.get("description"))
    label = raw
    if translated:
        label = f"{raw} -> {translated}" if raw else translated
    details = []
    if typ:
        details.append(typ)
    if gender:
        details.append(gender)
    if desc:
        details.append(desc)
    suffix = f" ({'; '.join(details)})" if details else ""
    if skipped:
        return f"{prefix}{label}{suffix}"
    return f"{prefix}{label}{suffix}"


def _chapter_metadata(progress_data, chapter):
    ci = int(chapter.get("chapter_index", 0))
    info = chapter.get("progress_entry") if isinstance(chapter, dict) else {}
    if not isinstance(info, dict):
        info = {}
    chapters = progress_data.get("chapters", {}) if isinstance(progress_data, dict) else {}
    if not info and isinstance(chapters, dict):
        info = chapters.get(str(ci)) or chapters.get(ci) or {}
    if not isinstance(info, dict):
        info = {}
    spine_number = chapter.get("spine_number") or ci + 1
    chapter_num = (
        chapter.get("progress_chapter_num")
        or chapter.get("display_chapter_num")
        or info.get("chapter_num")
        or info.get("actual_num")
        or chapter.get("chapter_num")
        or ci + 1
    )
    filename = (
        chapter.get("output_file")
        or info.get("output_file")
        or info.get("chapter_file")
        or chapter.get("filename")
        or f"chapter {ci + 1}"
    )
    model = (
        chapter.get("model_name")
        or chapter.get("model")
        or info.get("model_name")
        or info.get("model")
        or (progress_data.get("model_name", "") if isinstance(progress_data, dict) else "")
    )
    return {
        "chapter_index": ci,
        "spine_number": spine_number,
        "chapter_num": chapter_num,
        "filename": filename,
        "model": model or "(model unknown)",
        "status": str(info.get("status", "") or "").lower(),
    }


def build_chapter_footnote(
    entries,
    chapter,
    progress_data=None,
    include_unavailable_note=True,
    output_dir=None,
    output_text=None,
    output_available=None,
    skip_unmatched_entries=False,
):
    progress_data = progress_data if isinstance(progress_data, dict) else {}
    matches = match_entries_for_text(entries, chapter.get("text", ""))
    output_path = ""
    if output_available is None:
        if output_text is not None:
            output_available = True
        elif output_dir:
            output_text, output_path = chapter_output_text(output_dir, chapter, progress_data)
            output_available = output_text is not None
        else:
            output_available = False
    meta = _chapter_metadata(progress_data, chapter)
    lines = [
        f"## Chapter {meta['chapter_num']} Glossary Footnote",
        "",
        f"- Spine: {meta['spine_number']}",
        f"- Progress chapter: {meta['chapter_num']}",
        f"- File: {meta['filename']}",
        f"- Model: {meta['model']}",
        "",
        "### Matched glossary entries",
    ]
    if not matches:
        lines.append("- No glossary entries matched this source chapter.")
    else:
        entry_lines = []
        skipped_unmatched_count = 0
        for entry in matches:
            if output_available:
                confirmed = entry_matches_translated_output(entry, output_text or "")
                skipped = not confirmed
            else:
                confirmed = False
                skipped = True
            if skip_unmatched_entries and skipped:
                skipped_unmatched_count += 1
                continue
            status_order = 0 if confirmed else 1
            entry_lines.append((status_order, len(entry_lines), _format_entry_line(entry, skipped=skipped, confirmed=confirmed)))
        if entry_lines:
            lines.extend(line for _status, _idx, line in sorted(entry_lines))
        elif skipped_unmatched_count:
            lines.append("- No matched glossary entries remained after skipping unmatched entries.")
    if include_unavailable_note and not output_available:
        detail = f" ({output_path})" if output_path else ""
        lines.extend(
            [
                "",
                f"- {WARNING_PREFIX} Output-file usage check is unavailable{detail}; resolve the translated output file before trusting entry status.",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def completed_chapter_indices(progress_data):
    if not isinstance(progress_data, dict):
        return []
    result = set()
    chapters = progress_data.get("chapters", {})
    if isinstance(chapters, dict):
        for key, info in chapters.items():
            if not isinstance(info, dict):
                continue
            if str(info.get("status", "")).lower() == "completed":
                try:
                    result.add(int(info.get("chapter_index", key)))
                except (TypeError, ValueError):
                    pass
    for value in progress_data.get("completed", []) if isinstance(progress_data.get("completed", []), list) else []:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            pass
    merged = set()
    for value in progress_data.get("merged_indices", []) if isinstance(progress_data.get("merged_indices", []), list) else []:
        try:
            merged.add(int(value))
        except (TypeError, ValueError):
            pass
    failed = set()
    for value in progress_data.get("failed", []) if isinstance(progress_data.get("failed", []), list) else []:
        try:
            failed.add(int(value))
        except (TypeError, ValueError):
            pass
    return sorted(result - merged - failed)


def build_completed_summary(entries, chapters, progress_data=None, title=None, output_dir=None):
    progress_data = progress_data if isinstance(progress_data, dict) else {}
    chapter_by_index = {int(ch.get("chapter_index", idx)): ch for idx, ch in enumerate(chapters)}
    indices = completed_chapter_indices(progress_data)
    title = title or progress_data.get("book_title") or "Glossary Footnotes"
    lines = [
        f"# {title} - Completed Glossary Footnotes",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    if not indices:
        lines.append("No completed glossary-progress chapters were found.")
        return "\n".join(lines).rstrip() + "\n"
    for ci in indices:
        chapter = chapter_by_index.get(ci)
        if not chapter:
            continue
        lines.append(
            build_chapter_footnote(
                entries,
                chapter,
                progress_data,
                include_unavailable_note=False,
                output_dir=output_dir,
            ).rstrip()
        )
        lines.append("")
    if not any(line.startswith("## ") for line in lines):
        lines.append("No completed chapters could be mapped back to source EPUB chapters.")
    return "\n".join(lines).rstrip() + "\n"


def write_completed_summary(entries, chapters, progress_data, output_dir, book_base):
    safe_base = re.sub(r'[<>:"/\\\\|?*]+', "_", str(book_base or "book")).strip(" ._") or "book"
    folder = os.path.join(output_dir or os.getcwd(), "glossary_footnotes")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{safe_base}_completed_glossary_footnotes.md")
    content = build_completed_summary(
        entries,
        chapters,
        progress_data,
        title=progress_data.get("book_title") if isinstance(progress_data, dict) else None,
        output_dir=output_dir,
    )
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return path, content


def compact_extracted_entries(entries):
    compact = []
    seen = set()
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        normalized = _normalize_entry(entry, len(compact))
        identity = normalized.get("identity")
        if not normalized.get("raw_name") or identity in seen:
            continue
        seen.add(identity)
        compact.append(
            {
                "type": normalized.get("type", ""),
                "raw_name": normalized.get("raw_name", ""),
                "translated_name": normalized.get("translated_name", ""),
                "gender": normalized.get("gender", ""),
            }
        )
    return compact
