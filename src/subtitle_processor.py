"""Structured SRT/ASS subtitle extraction and round-trip writing.

Only subtitle dialogue is sent for translation.  Cue timing, numbering, ASS
headers/styles/event fields, line endings, and inline formatting tokens remain
in the original template and are restored locally.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple


SUBTITLE_EXTENSIONS = {".srt", ".ass"}
SUBTITLE_PROMPT_PROFILE_NAME = "Subtitle Translation"
DEFAULT_SUBTITLE_TRANSLATION_PROMPT = (
    "You are a professional subtitle translator. Translate every source value "
    "to {target_lang}.\n"
    "- Write concise, natural spoken dialogue suitable for on-screen subtitles.\n"
    "- Preserve the complete meaning, tone, emotion, humor, relationships, "
    "character voice, and level of formality without unnecessary expansion.\n"
    "- Do not summarize, censor, embellish, explain, or add translator notes.\n"
    "- Do not add speaker names, labels, punctuation, or context that is not "
    "present in the source.\n"
    "- Preserve meaningful line breaks and keep each cue independently usable; "
    "do not merge, split, omit, or reorder cues.\n"
    "- Input is a JSON array of objects with id and source fields. Output only "
    "a valid JSON array whose objects contain exactly id and target fields.\n"
    "- Preserve every id exactly once and in the same order.\n"
    "- Preserve every placeholder exactly as written, including tokens such as "
    "[[SUB_TAG_000001_0000]]. Never add, remove, duplicate, reorder, or "
    "translate placeholders.\n"
    "- Preserve variables, inline formatting markers, and meaningful line "
    "breaks. Output no markdown fences, explanations, or extra fields.\n"
)
SUBTITLE_PLACEHOLDER_RE = re.compile(r"\[\[SUB_TAG_\d{6}_\d{4}\]\]")
_SUBTITLE_TOKEN_COUNTERS = threading.local()
_SRT_TIMESTAMP_RE = re.compile(
    # Real-world SRT exports do not always zero-pad fractional seconds to
    # three digits (for example 00:00:07,0). Preserve that syntax verbatim
    # while still recognizing the cue as translatable dialogue.
    r"^\s*\d{1,3}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*"
    r"\d{1,3}:\d{2}:\d{2}[,.]\d{1,3}(?:\s+.*)?$"
)
_PROTECTED_TOKEN_RE = re.compile(
    r"<[^>\r\n]+>|\{[^{}\r\n]*\}|\\[Nnh]",
    re.IGNORECASE,
)
_ASS_DRAWING_MODE_RE = re.compile(r"\\p(\d+)", re.IGNORECASE)
_UNSAFE_ARCHIVE_COMPONENT_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class SubtitleArchiveError(ValueError):
    """Raised when a subtitle archive is unsafe or exceeds extraction limits."""


def is_subtitle_path(path: str) -> bool:
    return os.path.splitext(str(path or ""))[1].lower() in SUBTITLE_EXTENSIONS


def _safe_archive_component(value: str, fallback: str) -> str:
    cleaned = _UNSAFE_ARCHIVE_COMPONENT_RE.sub("_", str(value or ""))
    cleaned = cleaned.strip().rstrip(" .")
    return cleaned or fallback


def _unique_archive_target(path: str, used: set) -> str:
    candidate = path
    stem, extension = os.path.splitext(path)
    counter = 2
    normalized = os.path.normcase(os.path.abspath(candidate))
    while normalized in used or os.path.exists(candidate):
        candidate = f"{stem}_{counter}{extension}"
        normalized = os.path.normcase(os.path.abspath(candidate))
        counter += 1
    used.add(normalized)
    return candidate


def plan_subtitle_archive_outputs(
    archive_path: str,
    subtitle_paths: Iterable[str],
    output_base_dir: str,
    work_base_dir: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """Map extracted subtitles to one archive-named output directory."""
    archive_stem = os.path.splitext(os.path.basename(str(archive_path or "")))[0]
    group_name = _safe_archive_component(archive_stem, "Subtitles")
    group_dir = os.path.abspath(os.path.join(output_base_dir, group_name))
    used_output_names = set()
    plan: Dict[str, Dict[str, str]] = {}

    for subtitle_path in subtitle_paths or []:
        source_path = os.path.abspath(str(subtitle_path))
        source_name = os.path.basename(source_path)
        source_stem, source_extension = os.path.splitext(source_name)
        safe_stem = _safe_archive_component(source_stem, "subtitle")
        extension = (
            source_extension
            if source_extension.lower() in SUBTITLE_EXTENSIONS
            else ".srt"
        )
        candidate = f"{safe_stem}{extension}"
        counter = 2
        while candidate.casefold() in used_output_names:
            candidate = f"{safe_stem}_{counter}{extension}"
            counter += 1
        used_output_names.add(candidate.casefold())
        final_output = os.path.join(group_dir, candidate)
        work_root = os.path.abspath(
            str(work_base_dir)
            if work_base_dir
            else os.path.join(group_dir, ".glossarion_subtitle_work")
        )
        stable_key = hashlib.sha256(
            os.path.normcase(final_output).encode("utf-8")
        ).hexdigest()[:12]
        plan[os.path.normcase(source_path)] = {
            "archive_path": os.path.abspath(str(archive_path)),
            "group_name": group_name,
            "output_dir": group_dir,
            "output_path": final_output,
            "work_dir": os.path.join(
                work_root,
                f"{safe_stem}_{stable_key}",
            ),
        }
    return plan


def grouped_subtitle_output_layout(
    source_path: str,
    output_group_dir: str,
    output_path: str,
    work_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Return isolated work and final paths for one grouped subtitle."""
    raw_group_dir = str(output_group_dir or "").strip()
    raw_output_path = str(output_path or "").strip()
    if not raw_group_dir or not raw_output_path:
        raise ValueError("Grouped subtitle output paths cannot be empty")
    group_dir = os.path.abspath(raw_group_dir)
    final_output = os.path.abspath(raw_output_path)
    try:
        if os.path.commonpath((group_dir, final_output)) != group_dir:
            raise ValueError(
                "Grouped subtitle output file must stay inside its archive folder"
            )
    except ValueError as exc:
        raise ValueError(
            "Grouped subtitle output file must stay inside its archive folder"
        ) from exc

    raw_work_dir = str(work_dir or "").strip()
    if raw_work_dir:
        resolved_work_dir = os.path.abspath(raw_work_dir)
    else:
        final_stem = os.path.splitext(os.path.basename(final_output))[0]
        safe_work_stem = _safe_archive_component(final_stem, "subtitle")
        stable_key = hashlib.sha256(
            os.path.normcase(final_output).encode("utf-8")
        ).hexdigest()[:12]
        resolved_work_dir = os.path.join(
            group_dir,
            ".glossarion_subtitle_work",
            f"{safe_work_stem}_{stable_key}",
        )
    return {
        "source_path": os.path.abspath(str(source_path)),
        "output_group_dir": group_dir,
        "output_path": final_output,
        "work_dir": resolved_work_dir,
    }


def extract_subtitle_archive(
    archive_path: str,
    extraction_root: str,
    *,
    max_files: int = 5000,
    max_single_file_bytes: int = 32 * 1024 * 1024,
    max_total_bytes: int = 256 * 1024 * 1024,
) -> Dict[str, Any]:
    """Safely extract SRT/ASS members from a ZIP without using extractall."""
    os.makedirs(extraction_root, exist_ok=True)
    root = os.path.abspath(extraction_root)
    selected: List[Tuple[zipfile.ZipInfo, List[str]]] = []
    ignored = 0
    declared_total = 0

    try:
        archive = zipfile.ZipFile(archive_path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise SubtitleArchiveError(f"Invalid ZIP archive: {exc}") from exc

    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            archive_name = str(info.filename or "").replace("\\", "/")
            pure_path = PurePosixPath(archive_name)
            parts = list(pure_path.parts)
            extension = os.path.splitext(parts[-1] if parts else "")[1].lower()
            if extension not in SUBTITLE_EXTENSIONS:
                ignored += 1
                continue
            if (
                not parts
                or pure_path.is_absolute()
                or re.match(r"^[A-Za-z]:", archive_name)
                or any(part in ("", ".", "..") for part in parts)
            ):
                raise SubtitleArchiveError(
                    f"Unsafe subtitle member path: {archive_name}"
                )
            unix_mode = (int(info.external_attr) >> 16) & 0xFFFF
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise SubtitleArchiveError(
                    f"Symlink subtitle members are not allowed: {archive_name}"
                )
            if info.flag_bits & 0x1:
                raise SubtitleArchiveError(
                    f"Encrypted subtitle members are not supported: {archive_name}"
                )
            if int(info.file_size) > max_single_file_bytes:
                raise SubtitleArchiveError(
                    f"Subtitle member is too large: {archive_name}"
                )
            declared_total += max(0, int(info.file_size))
            if declared_total > max_total_bytes:
                raise SubtitleArchiveError(
                    "Subtitle archive exceeds the safe extraction size limit"
                )
            if len(selected) >= max_files:
                raise SubtitleArchiveError(
                    "Subtitle archive contains too many subtitle files"
                )
            safe_parts = [
                _safe_archive_component(part, f"folder_{index + 1}")
                for index, part in enumerate(parts)
            ]
            selected.append((info, safe_parts))

        extracted: List[str] = []
        used_targets: set = set()
        actual_total = 0
        for info, safe_parts in selected:
            target = os.path.abspath(os.path.join(root, *safe_parts))
            try:
                if os.path.commonpath((root, target)) != root:
                    raise SubtitleArchiveError(
                        f"Unsafe subtitle extraction target: {info.filename}"
                    )
            except ValueError as exc:
                raise SubtitleArchiveError(
                    f"Unsafe subtitle extraction target: {info.filename}"
                ) from exc
            os.makedirs(os.path.dirname(target), exist_ok=True)
            target = _unique_archive_target(target, used_targets)
            written = 0
            try:
                with archive.open(info, "r") as source, open(target, "wb") as destination:
                    while True:
                        chunk = source.read(min(1024 * 1024, max_single_file_bytes + 1 - written))
                        if not chunk:
                            break
                        written += len(chunk)
                        actual_total += len(chunk)
                        if (
                            written > max_single_file_bytes
                            or actual_total > max_total_bytes
                        ):
                            raise SubtitleArchiveError(
                                "Subtitle archive exceeded its safe extraction limit"
                            )
                        destination.write(chunk)
            except Exception:
                try:
                    os.remove(target)
                except OSError:
                    pass
                raise
            extracted.append(target)

    return {
        "archive_path": os.path.abspath(archive_path),
        "extraction_root": root,
        "files": extracted,
        "subtitle_count": len(extracted),
        "ignored_count": ignored,
        "total_bytes": sum(os.path.getsize(path) for path in extracted),
    }


def _decode_subtitle(path: str) -> Tuple[str, str]:
    with open(path, "rb") as source:
        data = source.read()

    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig"), "utf-8-sig"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16"), "utf-16"

    for encoding in ("utf-8", "cp1252"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"


def _line_records(text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    position = 0
    length = len(text)
    while position < length:
        newline_match = re.search(r"\r\n|\n|\r", text[position:])
        if newline_match is None:
            end = length
            content_end = length
        else:
            content_end = position + newline_match.start()
            end = position + newline_match.end()
        records.append(
            {
                "start": position,
                "content_end": content_end,
                "end": end,
                "text": text[position:content_end],
            }
        )
        position = end
    if not records and text == "":
        return []
    return records


def _next_placeholder(segment_index: int, counter: int) -> str:
    return f"[[SUB_TAG_{segment_index:06d}_{counter:04d}]]"


def _protect_subtitle_text(
    text: str, segment_index: int
) -> Tuple[str, List[Dict[str, str]]]:
    """Mask markup, ASS override blocks, escapes, and drawing commands."""
    protected: List[Dict[str, str]] = []
    output: List[str] = []
    cursor = 0
    drawing_mode = 0

    def add_token(value: str) -> None:
        token = _next_placeholder(segment_index, len(protected))
        protected.append({"token": token, "value": value})
        output.append(token)

    for match in _PROTECTED_TOKEN_RE.finditer(text):
        between = text[cursor : match.start()]
        if between:
            if drawing_mode:
                add_token(between)
            else:
                output.append(between)

        value = match.group(0)
        add_token(value)
        if value.startswith("{"):
            modes = _ASS_DRAWING_MODE_RE.findall(value)
            if modes:
                drawing_mode = int(modes[-1])
        cursor = match.end()

    tail = text[cursor:]
    if tail:
        if drawing_mode:
            add_token(tail)
        else:
            output.append(tail)

    return "".join(output), protected


def _has_translatable_text(masked_text: str) -> bool:
    visible = SUBTITLE_PLACEHOLDER_RE.sub("", str(masked_text or ""))
    return bool(visible.strip())


def _extract_srt_spans(text: str) -> List[Tuple[int, int, str]]:
    lines = _line_records(text)
    spans: List[Tuple[int, int, str]] = []
    timestamp_indices = [
        index
        for index, line in enumerate(lines)
        if _SRT_TIMESTAMP_RE.match(str(line["text"]))
    ]
    for cue_index, timestamp_index in enumerate(timestamp_indices):
        next_timestamp = (
            timestamp_indices[cue_index + 1]
            if cue_index + 1 < len(timestamp_indices)
            else len(lines)
        )
        dialogue = list(lines[timestamp_index + 1 : next_timestamp])
        while dialogue and not str(dialogue[-1]["text"]).strip():
            dialogue.pop()
        # Tolerate SRT files that omit the blank separator: the cue number
        # immediately before the next timestamp still belongs to the next cue.
        if (
            cue_index + 1 < len(timestamp_indices)
            and dialogue
            and re.fullmatch(r"\s*\d+\s*", str(dialogue[-1]["text"]))
        ):
            dialogue.pop()
            while dialogue and not str(dialogue[-1]["text"]).strip():
                dialogue.pop()
        while dialogue and not str(dialogue[0]["text"]).strip():
            dialogue.pop(0)
        if not dialogue:
            continue
        start = int(dialogue[0]["start"])
        end = int(dialogue[-1]["content_end"])
        if end > start:
            spans.append((start, end, text[start:end]))
    return spans


def _ass_text_offset(payload: str, field_count: int) -> Optional[int]:
    if field_count <= 1:
        return 0
    cursor = 0
    for _ in range(field_count - 1):
        comma = payload.find(",", cursor)
        if comma < 0:
            return None
        cursor = comma + 1
    return cursor


def _extract_ass_spans(text: str) -> List[Tuple[int, int, str]]:
    lines = _line_records(text)
    spans: List[Tuple[int, int, str]] = []
    in_events = False
    event_format: Optional[List[str]] = None

    for line in lines:
        line_text = str(line["text"])
        section_match = re.match(r"^\s*\[([^\]]+)\]\s*$", line_text)
        if section_match:
            in_events = section_match.group(1).strip().casefold() == "events"
            event_format = None
            continue
        if not in_events:
            continue

        format_match = re.match(r"^\s*Format\s*:\s*(.*)$", line_text, re.IGNORECASE)
        if format_match:
            event_format = [
                field.strip().casefold()
                for field in format_match.group(1).split(",")
            ]
            continue

        dialogue_match = re.match(
            r"^(\s*Dialogue\s*:\s*)(.*)$", line_text, re.IGNORECASE
        )
        if not dialogue_match:
            continue

        fields = event_format or [
            "layer",
            "start",
            "end",
            "style",
            "name",
            "marginl",
            "marginr",
            "marginv",
            "effect",
            "text",
        ]
        try:
            text_index = fields.index("text")
        except ValueError:
            continue
        # The ASS/SSA specification places Text last, allowing dialogue text
        # itself to contain commas without escaping.
        if text_index != len(fields) - 1:
            continue

        prefix = dialogue_match.group(1)
        payload = dialogue_match.group(2)
        payload_text_offset = _ass_text_offset(payload, len(fields))
        if payload_text_offset is None:
            continue
        local_start = len(prefix) + payload_text_offset
        start = int(line["start"]) + local_start
        end = int(line["content_end"])
        if end > start:
            spans.append((start, end, text[start:end]))

    return spans


def _count_tokens(text: str) -> int:
    try:
        from chapter_splitter import ChapterSplitter

        model_name = os.getenv("MODEL", "gpt-3.5-turbo")
        counters = getattr(_SUBTITLE_TOKEN_COUNTERS, "by_model", None)
        if counters is None:
            counters = {}
            _SUBTITLE_TOKEN_COUNTERS.by_model = counters
        counter = counters.get(model_name)
        if counter is None:
            counter = ChapterSplitter(model_name=model_name)
            counters[model_name] = counter
        return counter.count_tokens(text)
    except Exception:
        return max(1, len(text) // 4)


def _available_tokens() -> int:
    try:
        explicit = int(os.getenv("SUBTITLE_AVAILABLE_TOKENS", "0") or "0")
    except (TypeError, ValueError):
        explicit = 0
    if explicit > 0:
        return max(1000, explicit)
    try:
        output_limit = int(os.getenv("MAX_OUTPUT_TOKENS", "8192") or "8192")
    except (TypeError, ValueError):
        output_limit = 8192
    try:
        compression = float(os.getenv("COMPRESSION_FACTOR", "2.0") or "2.0")
    except (TypeError, ValueError):
        compression = 2.0
    if compression <= 0:
        compression = 1.0
    return max(1000, int((output_limit - 500) / compression))


def _batch_body(segments: List[Dict[str, Any]]) -> str:
    records = [
        {"id": str(segment["id"]), "source": str(segment["source_text"])}
        for segment in segments
    ]
    return json.dumps(records, ensure_ascii=False, indent=2)


def _pack_batches(
    segments: List[Dict[str, Any]], available_tokens: int
) -> List[List[Dict[str, Any]]]:
    """Pack the largest exact token-fitting slices without quadratic rescans."""
    batches: List[List[Dict[str, Any]]] = []
    start = 0
    segment_count = len(segments)
    while start < segment_count:
        # A single oversized cue must still make forward progress.
        low = start + 1
        high = segment_count
        best_end = low
        while low <= high:
            candidate_end = (low + high) // 2
            candidate_tokens = _count_tokens(
                _batch_body(segments[start:candidate_end])
            )
            if candidate_tokens <= available_tokens:
                best_end = candidate_end
                low = candidate_end + 1
            else:
                high = candidate_end - 1
        batches.append(segments[start:best_end])
        start = best_end
    return batches


def _subtitle_extraction_worker_count(source_count: int) -> int:
    if source_count <= 1:
        return 1
    try:
        configured = int(
            os.getenv("SUBTITLE_EXTRACTION_WORKERS", "0") or "0"
        )
    except (TypeError, ValueError):
        configured = 0
    if configured <= 0:
        configured = min(8, max(2, int(os.cpu_count() or 4)))
    return max(1, min(int(source_count), configured))


def extract_subtitle_to_chapters(path: str, output_dir: str) -> Dict[str, Any]:
    extension = os.path.splitext(path)[1].lower()
    if extension not in SUBTITLE_EXTENSIONS:
        raise ValueError(f"Unsupported subtitle extension: {extension}")

    os.makedirs(output_dir, exist_ok=True)
    source_text, source_encoding = _decode_subtitle(path)
    raw_spans = (
        _extract_srt_spans(source_text)
        if extension == ".srt"
        else _extract_ass_spans(source_text)
    )

    segments: List[Dict[str, Any]] = []
    for start, end, original_text in raw_spans:
        segment_index = len(segments) + 1
        masked_text, protected = _protect_subtitle_text(
            original_text, segment_index
        )
        if not _has_translatable_text(masked_text):
            continue
        segments.append(
            {
                "id": str(segment_index),
                "start": start,
                "end": end,
                "original_text": original_text,
                "source_text": masked_text,
                "placeholders": protected,
            }
        )

    available = _available_tokens()
    batches = _pack_batches(segments, available)
    chapters: List[Dict[str, Any]] = []
    manifest_batches: List[Dict[str, Any]] = []
    for batch_num, batch_segments in enumerate(batches, start=1):
        body = _batch_body(batch_segments)
        filename = f"section_{batch_num}.txt"
        segment_ids = [str(segment["id"]) for segment in batch_segments]
        chapters.append(
            {
                "num": batch_num,
                "title": f"Subtitle Batch {batch_num}",
                "body": body,
                "filename": filename,
                "source_file": os.path.abspath(path),
                "original_basename": os.path.basename(path),
                "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "file_size": len(body),
                "has_images": False,
                "image_count": 0,
                "is_chunk": False,
                "subtitle_batch": True,
                "structured_translation_batch": True,
                "structured_batch_kind": "subtitle",
                "subtitle_segment_ids": segment_ids,
                "subtitle_source_batch_num": batch_num,
                "subtitle_source_batch_count": len(batches),
                "subtitle_progress_id": os.path.abspath(path),
            }
        )
        manifest_batches.append(
            {
                "num": batch_num,
                "filename": filename,
                "segment_ids": segment_ids,
            }
        )

    source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    manifest = {
        "version": 1,
        "type": "subtitle_json_batches",
        "format": extension.lstrip("."),
        "source_file": os.path.abspath(path),
        "source_basename": os.path.basename(path),
        "source_encoding": source_encoding,
        "source_hash": source_hash,
        "segment_count": len(segments),
        "batch_count": len(chapters),
        "batches": manifest_batches,
        "segments": segments,
    }
    metadata = {
        "title": os.path.splitext(os.path.basename(path))[0],
        "type": "subtitle",
        "format": extension.lstrip("."),
        "source_file": os.path.abspath(path),
        "source_encoding": source_encoding,
        "chapter_count": len(chapters),
        "segment_count": len(segments),
        "batch_count": len(chapters),
    }

    manifest_path = os.path.join(output_dir, "subtitle_manifest.json")
    chapters_path = os.path.join(output_dir, "chapters_full.json")
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(manifest_path, "w", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
    with open(chapters_path, "w", encoding="utf-8") as target:
        json.dump(chapters, target, ensure_ascii=False)
    with open(metadata_path, "w", encoding="utf-8") as target:
        json.dump(metadata, target, ensure_ascii=False, indent=2)

    return {
        "success": True,
        "chapters": len(chapters),
        "segments": len(segments),
        "chapters_path": chapters_path,
        "manifest_path": manifest_path,
        "metadata": metadata,
        "empty_sources": (
            [
                {
                    "source_index": 1,
                    "source_file": os.path.abspath(path),
                    "output_path": None,
                    "source_hash": source_hash,
                    "bundle": False,
                }
            ]
            if not chapters
            else []
        ),
    }


def extract_subtitle_bundle_to_chapters(
    paths: Iterable[str],
    output_dir: str,
    output_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Create one chapter list for every subtitle file in an archive bundle."""
    source_paths = [os.path.abspath(str(path)) for path in (paths or [])]
    if not source_paths:
        raise ValueError("Subtitle bundle contains no input files")

    os.makedirs(output_dir, exist_ok=True)
    source_manifest_root = os.path.join(output_dir, ".source_manifests")
    normalized_outputs = {
        os.path.normcase(os.path.abspath(str(source_path))): os.path.abspath(
            str(target_path)
        )
        for source_path, target_path in (output_paths or {}).items()
        if source_path and target_path
    }
    chapters: List[Dict[str, Any]] = []
    bundle_files: List[Dict[str, Any]] = []
    empty_sources: List[Dict[str, Any]] = []
    total_segments = 0

    def _extract_bundle_source(source_item):
        source_index, source_path = source_item
        source_dir = os.path.join(source_manifest_root, f"source_{source_index:05d}")
        result = extract_subtitle_to_chapters(source_path, source_dir)
        return source_index, source_path, result

    indexed_sources = list(enumerate(source_paths, start=1))
    extraction_workers = _subtitle_extraction_worker_count(len(indexed_sources))
    if extraction_workers > 1:
        print(
            f"Preparing {len(indexed_sources)} subtitle files with "
            f"{extraction_workers} parallel extraction workers"
        )
        with ThreadPoolExecutor(
            max_workers=extraction_workers,
            thread_name_prefix="SubtitleExtract",
        ) as executor:
            extracted_sources = list(
                executor.map(_extract_bundle_source, indexed_sources)
            )
    else:
        extracted_sources = [
            _extract_bundle_source(source_item)
            for source_item in indexed_sources
        ]

    for source_index, source_path, result in extracted_sources:
        with open(result["manifest_path"], "r", encoding="utf-8") as source:
            manifest = json.load(source)
        with open(result["chapters_path"], "r", encoding="utf-8") as source:
            source_chapters = json.load(source)

        source_batches = manifest.get("batches", [])
        if (
            not isinstance(source_batches, list)
            or not isinstance(source_chapters, list)
            or len(source_batches) != len(source_chapters)
        ):
            raise ValueError(
                f"Subtitle batch manifest mismatch for {source_path}"
            )
        output_path = normalized_outputs.get(os.path.normcase(source_path))
        if not output_path:
            source_stem, source_extension = os.path.splitext(
                os.path.basename(source_path)
            )
            output_path = os.path.join(
                output_dir,
                f"{source_stem}{source_extension or '.srt'}",
            )
        updated_batches = []
        for local_batch, chapter in zip(
            source_batches,
            source_chapters,
        ):
            global_batch_num = len(chapters) + 1
            global_filename = f"section_{global_batch_num}.txt"
            updated_batch = dict(local_batch)
            updated_batch["num"] = global_batch_num
            updated_batch["filename"] = global_filename
            updated_batch["source_batch_num"] = int(
                local_batch.get("num") or len(updated_batches) + 1
            )
            updated_batch["source_batch_count"] = len(source_batches)
            updated_batches.append(updated_batch)

            updated_chapter = dict(chapter)
            updated_chapter["num"] = global_batch_num
            updated_chapter["filename"] = global_filename
            updated_chapter["title"] = (
                f"{os.path.basename(source_path)} — "
                f"Subtitle Batch {local_batch.get('num', len(updated_batches))}"
            )
            updated_chapter["subtitle_bundle"] = True
            updated_chapter["subtitle_bundle_source_index"] = source_index
            updated_chapter["subtitle_bundle_source_file"] = source_path
            updated_chapter["subtitle_progress_id"] = output_path
            updated_chapter["subtitle_output_file"] = output_path
            chapters.append(updated_chapter)

        manifest["batches"] = updated_batches
        manifest["batch_count"] = len(updated_batches)
        if not updated_batches:
            empty_sources.append(
                {
                    "source_index": source_index,
                    "source_file": source_path,
                    "output_path": output_path,
                    "source_hash": str(manifest.get("source_hash") or ""),
                    "bundle": True,
                }
            )
        bundle_files.append(
            {
                "source_file": source_path,
                "output_path": output_path,
                "manifest": manifest,
            }
        )
        total_segments += int(manifest.get("segment_count") or 0)

    metadata = {
        "title": "Subtitle Archive Bundle",
        "type": "subtitle_bundle",
        "source_files": source_paths,
        "source_count": len(source_paths),
        "chapter_count": len(chapters),
        "segment_count": total_segments,
        "batch_count": len(chapters),
    }
    bundle_manifest = {
        "version": 1,
        "type": "subtitle_archive_bundle",
        "files": bundle_files,
        "chapter_count": len(chapters),
        "segment_count": total_segments,
    }
    chapters_path = os.path.join(output_dir, "chapters_full.json")
    metadata_path = os.path.join(output_dir, "metadata.json")
    bundle_manifest_path = os.path.join(
        output_dir,
        "subtitle_bundle_manifest.json",
    )
    with open(chapters_path, "w", encoding="utf-8") as target:
        json.dump(chapters, target, ensure_ascii=False)
    with open(metadata_path, "w", encoding="utf-8") as target:
        json.dump(metadata, target, ensure_ascii=False, indent=2)
    with open(bundle_manifest_path, "w", encoding="utf-8") as target:
        json.dump(bundle_manifest, target, ensure_ascii=False, indent=2)

    return {
        "success": True,
        "chapters": len(chapters),
        "segments": total_segments,
        "source_count": len(source_paths),
        "chapters_path": chapters_path,
        "manifest_path": bundle_manifest_path,
        "metadata": metadata,
        "empty_sources": empty_sources,
    }


def _batch_output_candidates(
    output_dir: str,
    filename: str,
    source_file: Optional[str] = None,
    source_batch_num: Optional[int] = None,
    source_batch_count: Optional[int] = None,
) -> Iterable[str]:
    """Yield manifest and source-named checkpoint paths for one subtitle batch."""
    candidates = [filename]
    if filename and not os.path.basename(filename).startswith("response_"):
        candidates.append(f"response_{filename}")
    if source_file:
        source_stem = os.path.splitext(os.path.basename(source_file))[0]
        try:
            local_batch = int(source_batch_num or 1)
        except (TypeError, ValueError):
            local_batch = 1
        try:
            local_count = int(source_batch_count or 1)
        except (TypeError, ValueError):
            local_count = 1
        checkpoint_stem = (
            f"{source_stem}_batch_{local_batch}"
            if local_count > 1
            else source_stem
        )
        candidates.extend(
            [
                f"{checkpoint_stem}.txt",
                f"response_{checkpoint_stem}.txt",
            ]
        )
    yielded = set()
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = os.path.join(output_dir, candidate)
        normalized = os.path.normcase(os.path.abspath(candidate_path))
        if normalized in yielded:
            continue
        yielded.add(normalized)
        yield candidate_path


def _read_completed_batch(
    output_dir: str,
    batch: Dict[str, Any],
    source_file: Optional[str] = None,
    source_batch_num: Optional[int] = None,
    source_batch_count: Optional[int] = None,
) -> Optional[str]:
    for candidate in _batch_output_candidates(
        output_dir,
        str(batch.get("filename") or ""),
        source_file=source_file,
        source_batch_num=source_batch_num,
        source_batch_count=source_batch_count,
    ):
        if not os.path.isfile(candidate):
            continue
        with open(candidate, "r", encoding="utf-8") as source:
            return source.read()
    return None


def _parse_batch_translation(
    text: str, expected_ids: List[str]
) -> Dict[str, str]:
    payload = json.loads(str(text or ""))
    if not isinstance(payload, list):
        raise ValueError("Subtitle batch output must be a JSON array")
    translations: Dict[str, str] = {}
    seen_order: List[str] = []
    expected_set = set(expected_ids)
    for item in payload:
        if not isinstance(item, dict) or set(item.keys()) != {"id", "target"}:
            raise ValueError(
                "Subtitle batch records must contain exactly id and target"
            )
        segment_id = str(item.get("id"))
        if segment_id in translations or segment_id not in expected_set:
            raise ValueError(f"Invalid subtitle batch id: {segment_id}")
        target = item.get("target")
        if not isinstance(target, str):
            raise ValueError(f"Subtitle target must be text: {segment_id}")
        translations[segment_id] = target
        seen_order.append(segment_id)
    if seen_order != expected_ids:
        raise ValueError("Subtitle batch ids are missing or out of order")
    return translations


def _restore_placeholders(
    translation: str, segment: Dict[str, Any]
) -> Optional[str]:
    placeholder_entries = segment.get("placeholders") or []
    expected = [str(item.get("token") or "") for item in placeholder_entries]
    if SUBTITLE_PLACEHOLDER_RE.findall(translation or "") != expected:
        return None
    restored = str(translation or "")
    for item in placeholder_entries:
        restored = restored.replace(
            str(item.get("token") or ""), str(item.get("value") or "")
        )
    original = str(segment.get("original_text") or "")
    newline_match = re.search(r"\r\n|\n|\r", original)
    if newline_match:
        restored = re.sub(r"\r\n|\n|\r", newline_match.group(0), restored)
    return restored


def _write_subtitle(path: str, text: str, encoding: str) -> str:
    used_encoding = encoding or "utf-8"
    try:
        data = text.encode(used_encoding)
    except UnicodeEncodeError:
        used_encoding = "utf-8-sig" if encoding == "utf-8-sig" else "utf-8"
        data = text.encode(used_encoding)
    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(path, "wb") as target:
        target.write(data)
    return used_encoding


def convert_subtitle(
    output_dir: str,
    manifest_path: Optional[str] = None,
    output_path: Optional[str] = None,
    manifest_data: Optional[Dict[str, Any]] = None,
    require_complete: bool = True,
) -> Dict[str, Any]:
    if manifest_data is None:
        manifest_path = manifest_path or os.path.join(
            output_dir, "subtitle_manifest.json"
        )
        with open(manifest_path, "r", encoding="utf-8") as source:
            manifest = json.load(source)
    else:
        manifest = dict(manifest_data)

    source_file = str(manifest["source_file"])
    source_text, detected_encoding = _decode_subtitle(source_file)
    current_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    if current_hash != manifest.get("source_hash"):
        raise RuntimeError(
            "The subtitle source changed after extraction; refusing to apply "
            "translations to stale cue positions."
        )

    segments_by_id = {
        str(segment.get("id")): segment
        for segment in manifest.get("segments", [])
        if isinstance(segment, dict)
    }
    translated: Dict[str, str] = {}
    invalid_batches = 0
    missing = 0
    manifest_batches = manifest.get("batches", [])
    batch_count = len(manifest_batches)
    for batch_position, batch in enumerate(manifest_batches, start=1):
        expected_ids = [
            str(segment_id) for segment_id in batch.get("segment_ids", [])
        ]
        source_batch_num = batch.get("source_batch_num")
        if source_batch_num is None:
            source_batch_num = (
                batch_position
                if batch_count > 1
                else 1
            )
        batch_text = _read_completed_batch(
            output_dir,
            batch,
            source_file=source_file,
            source_batch_num=source_batch_num,
            source_batch_count=(
                batch.get("source_batch_count") or batch_count
            ),
        )
        if batch_text is None:
            missing += len(expected_ids)
            continue
        try:
            translated.update(
                _parse_batch_translation(batch_text, expected_ids)
            )
        except Exception:
            invalid_batches += 1
            missing += len(expected_ids)

    replacements: List[Tuple[int, int, str]] = []
    skipped = 0
    for segment_id, segment in segments_by_id.items():
        target = translated.get(segment_id)
        if target is None:
            skipped += 1
            continue
        restored = _restore_placeholders(target, segment)
        if restored is None:
            skipped += 1
            continue
        replacements.append(
            (int(segment["start"]), int(segment["end"]), restored)
        )

    if output_path is None:
        stem, extension = os.path.splitext(os.path.basename(source_file))
        output_path = os.path.join(
            output_dir, f"{stem}{extension or '.srt'}"
        )

    if require_complete and (missing or invalid_batches or skipped):
        return {
            "success": False,
            "ready": False,
            "created": False,
            "output_path": output_path,
            "updated": len(replacements),
            "skipped": skipped,
            "missing": missing,
            "invalid_batches": invalid_batches,
            "output_encoding": None,
        }

    rebuilt = source_text
    for start, end, replacement in sorted(
        replacements, key=lambda item: item[0], reverse=True
    ):
        rebuilt = rebuilt[:start] + replacement + rebuilt[end:]

    output_encoding = _write_subtitle(
        output_path,
        rebuilt,
        str(manifest.get("source_encoding") or detected_encoding),
    )
    return {
        "success": True,
        "ready": True,
        "created": True,
        "output_path": output_path,
        "updated": len(replacements),
        "skipped": skipped,
        "missing": missing,
        "invalid_batches": invalid_batches,
        "output_encoding": output_encoding,
    }


_BUNDLE_CONVERSION_LOCK = threading.RLock()
_BUNDLE_CONVERSION_SIGNATURES: Dict[
    Tuple[str, int], Tuple[Tuple[str, int, int], ...]
] = {}


def _completed_batch_signature(
    output_dir: str,
    manifest: Dict[str, Any],
) -> Optional[Tuple[Tuple[str, int, int], ...]]:
    """Identify the exact completed response files used for one subtitle."""
    signature: List[Tuple[str, int, int]] = []
    manifest_batches = manifest.get("batches", [])
    batch_count = len(manifest_batches)
    source_file = str(manifest.get("source_file") or "")
    for batch_position, batch in enumerate(manifest_batches, start=1):
        completed_path = None
        for candidate in _batch_output_candidates(
            output_dir,
            str(batch.get("filename") or ""),
            source_file=source_file,
            source_batch_num=(
                batch.get("source_batch_num")
                or (batch_position if batch_count > 1 else 1)
            ),
            source_batch_count=(
                batch.get("source_batch_count") or batch_count
            ),
        ):
            if not os.path.isfile(candidate):
                continue
            completed_path = candidate
            break
        if completed_path is None:
            return None
        file_stat = os.stat(completed_path)
        signature.append(
            (
                os.path.normcase(os.path.abspath(completed_path)),
                int(file_stat.st_mtime_ns),
                int(file_stat.st_size),
            )
        )
    return tuple(signature)


def convert_subtitle_bundle_source(
    output_dir: str,
    source_index: int,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Rebuild one bundle subtitle as soon as all of its batches are complete."""
    manifest_path = os.path.abspath(
        manifest_path
        or os.path.join(output_dir, "subtitle_bundle_manifest.json")
    )
    try:
        selected_index = int(source_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("Subtitle bundle source index must be an integer") from exc
    if selected_index < 1:
        raise ValueError("Subtitle bundle source index must be at least 1")

    with _BUNDLE_CONVERSION_LOCK:
        with open(manifest_path, "r", encoding="utf-8") as source:
            bundle_manifest = json.load(source)
        if bundle_manifest.get("type") != "subtitle_archive_bundle":
            raise ValueError("Invalid subtitle archive bundle manifest")
        files = bundle_manifest.get("files", [])
        if not isinstance(files, list) or selected_index > len(files):
            raise IndexError(
                f"Subtitle bundle source index {selected_index} is out of range"
            )
        item = files[selected_index - 1]
        if not isinstance(item, dict) or not isinstance(
            item.get("manifest"), dict
        ):
            raise ValueError(
                "Subtitle archive bundle contains an invalid file entry"
            )
        output_path = str(item.get("output_path") or "").strip()
        if not output_path:
            raise ValueError(
                "Subtitle archive bundle contains a file without an output path"
            )

        signature = _completed_batch_signature(
            output_dir,
            item["manifest"],
        )
        cache_key = (manifest_path, selected_index)
        if (
            signature is not None
            and _BUNDLE_CONVERSION_SIGNATURES.get(cache_key) == signature
            and os.path.isfile(output_path)
        ):
            return {
                "success": True,
                "ready": True,
                "created": False,
                "already_exists": True,
                "output_path": output_path,
                "updated": int(item["manifest"].get("segment_count") or 0),
                "skipped": 0,
                "missing": 0,
                "invalid_batches": 0,
            }

        result = convert_subtitle(
            output_dir,
            output_path=output_path,
            manifest_data=item["manifest"],
            require_complete=True,
        )
        result["source_index"] = selected_index
        result["source_file"] = item.get("source_file")
        if result.get("ready"):
            completed_signature = _completed_batch_signature(
                output_dir,
                item["manifest"],
            )
            if completed_signature is not None:
                _BUNDLE_CONVERSION_SIGNATURES[cache_key] = completed_signature
        return result


def convert_subtitle_bundle(
    output_dir: str,
    manifest_path: Optional[str] = None,
    reuse_existing_source_indices: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Rebuild every source subtitle represented by a combined bundle."""
    manifest_path = manifest_path or os.path.join(
        output_dir,
        "subtitle_bundle_manifest.json",
    )
    with open(manifest_path, "r", encoding="utf-8") as source:
        bundle_manifest = json.load(source)
    if bundle_manifest.get("type") != "subtitle_archive_bundle":
        raise ValueError("Invalid subtitle archive bundle manifest")

    bundle_files = bundle_manifest.get("files", [])
    if not isinstance(bundle_files, list):
        raise ValueError("Subtitle archive bundle files must be a list")
    reuse_indices = {
        int(source_index)
        for source_index in (reuse_existing_source_indices or [])
        if str(source_index).strip().isdigit()
    }
    results = []
    for source_index in range(1, len(bundle_files) + 1):
        item = bundle_files[source_index - 1]
        existing_output = (
            str(item.get("output_path") or "").strip()
            if isinstance(item, dict)
            else ""
        )
        if source_index in reuse_indices and os.path.isfile(existing_output):
            item_manifest = (
                item.get("manifest", {})
                if isinstance(item, dict)
                else {}
            )
            results.append(
                {
                    "success": True,
                    "ready": True,
                    "created": False,
                    "already_exists": True,
                    "output_path": existing_output,
                    "source_index": source_index,
                    "source_file": (
                        item.get("source_file")
                        if isinstance(item, dict)
                        else None
                    ),
                    "updated": int(item_manifest.get("segment_count") or 0),
                    "skipped": 0,
                    "missing": 0,
                    "invalid_batches": 0,
                }
            )
            continue
        results.append(
            convert_subtitle_bundle_source(
                output_dir,
                source_index,
                manifest_path=manifest_path,
            )
        )
    ready_results = [result for result in results if result.get("ready")]
    incomplete_results = [
        result for result in results if not result.get("ready")
    ]
    return {
        "success": not incomplete_results,
        "files": len(ready_results),
        "total_files": len(results),
        "incomplete_files": len(incomplete_results),
        "outputs": [
            result.get("output_path") for result in ready_results
        ],
        "updated": sum(int(result.get("updated") or 0) for result in results),
        "skipped": sum(int(result.get("skipped") or 0) for result in results),
        "missing": sum(int(result.get("missing") or 0) for result in results),
        "invalid_batches": sum(
            int(result.get("invalid_batches") or 0) for result in results
        ),
        "results": results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Write translated SRT/ASS from Glossarion JSON batches."
    )
    parser.add_argument("output_dir")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            convert_subtitle(args.output_dir, args.manifest, args.output),
            ensure_ascii=False,
        )
    )
