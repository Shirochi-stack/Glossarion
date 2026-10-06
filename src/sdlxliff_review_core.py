"""SDLXLIFF reviewer core: the GUI-free half of Retranslation_GUI's SDLXLIFF review (mobile milestone U7).

Moved verbatim from ``git show 41814faa:src/Retranslation_GUI.py`` (RG line numbers):

* module helpers RG 342-668, 714-745: sidecar identity, the freshness manifest
  (``SDLXLIFF/sdlxliff_manifest.json``), source records, the manual-editing output name,
  the Machine Translation preview path and the application folder;
* ``SdlxliffAutogenMixin``: RetranslationMixin's sidecar auto-generation (RG 14979-14990,
  15199-16254 except the "Text Analysis Unavailable" message): source lookup in the EPUB /
  extracted folders, spine order, ``_generate_sdlxliff_sidecars_from_completed_entries``,
  ``..._from_untranslated_entries`` and the sidecar freshness check;
* ``SdlxliffReviewCoreMixin``: every SDLXLIFFReviewDialog method that needs no widget (RG
  845-925 constants, 9378 ``_GT_LANG_CODES``, the methods listed in
  tests/test_sdlxliff_review_core.py): sidecar parsing, text units, source/output alignment and
  row status analysis, the refresh scan and sidecar regeneration, staleness, edit saving incl.
  Manual editing, the pending-edit queues (row and Notepad document), Mark as Completed / Undo
  (``translation_progress.json`` + ``SDLXLIFF/review_status_overrides.json``), the Machine
  Translation preview (``google_free_translate`` provider chain, credentials, the JSON cache),
  inject and Flag inaccurate with its threshold.  Widget updates are hooks with GUI-free
  defaults; ``SDLXLIFFReviewDialog(SdlxliffReviewCoreMixin, QDialog)`` overrides each hook
  with its original code, and its Qt halves (the list, pages, Notepad WebEngine JS, menus,
  prompts, threads) stay in Retranslation_GUI.

Mobile: ``SdlxliffReviewSession`` / ``open_sdlxliff_review`` (status label and save timer are
plain stand-ins).  Retranslation_GUI re-exports every moved name.  Rules: Python 3.10
compatible; never import PySide6, translator_gui or dpi_setup.
"""

import copy
import hashlib
import html as html_lib
import importlib.util
import json
import os
import platform
import re
import sys
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from difflib import SequenceMatcher
from urllib.parse import unquote

from chapter_display_numbering import nonreset_chapter_display_numbers
from epub_package import find_opf_path as find_workspace_opf_path
from glossary_usage import _is_special_basename, read_epub_spine_chapters
from progress_core import (
    _IS_MACOS,
    _PROGRESS_READER_HTML_EXTENSIONS,
    _normalize_progress_match_name,
    _write_progress_snapshot_atomic,
)
from sdlxliff_sidecar_writer import (
    _SIDECAR_FRESHNESS_MANIFEST_LOCK,
    _SIDECAR_FRESHNESS_MANIFEST_TYPE,
    USER_ADDED_BREAK_POSITIONS_ATTRIBUTE,
    USER_ADDED_TARGET_INDEXES_ATTRIBUTE,
    _blank_manual_untranslated_sdlxliff_target,
    _clear_manual_untranslated_sdlxliff,
    _is_manual_editing_sdlxliff,
    _is_manual_untranslated_sdlxliff,
    _write_html_sdlxliff_sidecar,
)

_MACHINE_TRANSLATION_DIR = "Machine_Translation"


def _sdlxliff_logical_output_key(path_or_name):
    """Return one identity for retained- and response-named sidecars."""
    name = os.path.basename(str(path_or_name or "").replace("\\", "/"))
    if name.casefold().endswith(".sdlxliff"):
        name = name[:-len(".sdlxliff")]
    return _normalize_progress_match_name(name).casefold()


def _existing_sdlxliff_sidecars_by_logical_output(output_dir):
    """Index sidecars without treating a filename-mode change as new work."""
    sidecar_dir = os.path.join(str(output_dir or ""), "SDLXLIFF")
    indexed = {}
    try:
        with os.scandir(sidecar_dir) as entries:
            for entry in entries:
                if not entry.is_file() or not entry.name.casefold().endswith(".sdlxliff"):
                    continue
                logical_key = _sdlxliff_logical_output_key(entry.name)
                if logical_key:
                    indexed.setdefault(logical_key, entry.path)
    except OSError:
        pass
    return indexed


_SDLXLIFF_SIDECAR_MANIFEST_VERSION = 1
_SDLXLIFF_SIDECAR_MANIFEST_TYPE = _SIDECAR_FRESHNESS_MANIFEST_TYPE
_SDLXLIFF_SIDECAR_MANIFEST_LOCK = _SIDECAR_FRESHNESS_MANIFEST_LOCK


def _sdlxliff_sidecar_manifest_path(output_dir):
    return os.path.join(
        str(output_dir or ""),
        "SDLXLIFF",
        "sdlxliff_manifest.json",
    )


def _read_sdlxliff_sidecar_manifest(output_dir):
    """Read the optional sidecar freshness manifest.

    ``None`` deliberately means "use the legacy mtime check".  This applies
    to missing and unreadable manifests so older workspaces remain usable.
    """
    manifest_path = _sdlxliff_sidecar_manifest_path(output_dir)
    if not os.path.isfile(manifest_path):
        return None
    try:
        with _SDLXLIFF_SIDECAR_MANIFEST_LOCK:
            with open(manifest_path, "r", encoding="utf-8") as source:
                manifest = json.load(source)
        if not isinstance(manifest, dict):
            return None
        if manifest.get("type") != _SDLXLIFF_SIDECAR_MANIFEST_TYPE:
            return None
        if not isinstance(manifest.get("entries"), dict):
            return None
        return manifest
    except Exception:
        return None


def _update_sdlxliff_sidecar_manifest(output_dir, entry_updates):
    """Merge successful sidecar hash records and replace the manifest atomically."""
    updates = {
        str(key): value
        for key, value in (entry_updates or {}).items()
        if key and isinstance(value, dict)
    }
    if not updates:
        return False
    manifest_path = _sdlxliff_sidecar_manifest_path(output_dir)
    try:
        with _SDLXLIFF_SIDECAR_MANIFEST_LOCK:
            current = _read_sdlxliff_sidecar_manifest(output_dir)
            if current is None:
                current = {
                    "version": _SDLXLIFF_SIDECAR_MANIFEST_VERSION,
                    "type": _SDLXLIFF_SIDECAR_MANIFEST_TYPE,
                    "hash_algorithm": "sha256",
                    "entries": {},
                }
            entries = current.setdefault("entries", {})
            entries.update(updates)
            current["version"] = _SDLXLIFF_SIDECAR_MANIFEST_VERSION
            current["type"] = _SDLXLIFF_SIDECAR_MANIFEST_TYPE
            current["hash_algorithm"] = "sha256"
            _write_progress_snapshot_atomic(manifest_path, current)
        return True
    except Exception:
        return False


def _sdlxliff_sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sdlxliff_sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sdlxliff_file_stat_record(path):
    stat = os.stat(path)
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(
            stat,
            "st_mtime_ns",
            int(stat.st_mtime * 1000000000),
        )),
    }


def _sdlxliff_source_record(source_path):
    """Describe the loose file or EPUB member used as the sidecar source."""
    location = str(source_path or "")
    if not location:
        return None
    container_path = ""
    member = ""
    if "!" in location:
        possible_container, possible_member = location.rsplit("!", 1)
        if os.path.isfile(possible_container):
            container_path = os.path.abspath(possible_container)
            member = possible_member
    if container_path:
        record = {
            "kind": "epub_member",
            "container": container_path,
            "member": member,
        }
        try:
            record.update(_sdlxliff_file_stat_record(container_path))
        except OSError:
            pass
        return record
    if os.path.isfile(location):
        file_path = os.path.abspath(location)
        record = {"kind": "file", "path": file_path}
        try:
            record.update(_sdlxliff_file_stat_record(file_path))
        except OSError:
            pass
        return record
    return {"kind": "unresolved", "location": location}


def _sdlxliff_decode_html_bytes(payload):
    for encoding in ("utf-8", "utf-8-sig", "cp949", "latin-1"):
        try:
            return payload.decode(encoding)
        except Exception:
            continue
    return payload.decode("utf-8", errors="replace")


def _sdlxliff_source_record_payload(source_record):
    if not isinstance(source_record, dict):
        return None
    kind = source_record.get("kind")
    try:
        if kind == "file":
            source_path = source_record.get("path")
            if not source_path or not os.path.isfile(source_path):
                return None
            with open(source_path, "rb") as source:
                return _sdlxliff_decode_html_bytes(source.read())
        if kind == "epub_member":
            container = source_record.get("container")
            member = source_record.get("member")
            if not container or not member or not os.path.isfile(container):
                return None
            with zipfile.ZipFile(container, "r") as source_zip:
                return _sdlxliff_decode_html_bytes(source_zip.read(member))
    except Exception:
        return None
    return None


def _sdlxliff_manifest_freshness_record(
    output_name,
    sidecar_path,
    output_path,
    source_html,
    source_path,
):
    output_stat = _sdlxliff_file_stat_record(output_path)
    source_record = _sdlxliff_source_record(source_path)
    return {
        "output_name": os.path.basename(str(output_name or "").replace("\\", "/")),
        "sidecar_name": os.path.basename(str(sidecar_path or "").replace("\\", "/")),
        "output_sha256": _sdlxliff_sha256_file(output_path),
        "output_size": output_stat["size"],
        "output_mtime_ns": output_stat["mtime_ns"],
        "source_sha256": _sdlxliff_sha256_bytes(str(source_html or "").encode("utf-8")),
        "source": source_record,
    }


def _sdlxliff_manifest_record_current(
    record,
    output_path,
    output_stat=None,
    source_stat_cache=None,
):
    """Return ``(is_current, refreshed_record)`` for a usable hash record."""
    if not isinstance(record, dict):
        return None, None
    expected_output_hash = str(record.get("output_sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_output_hash):
        return None, None
    if not isinstance(output_stat, dict):
        try:
            output_stat = _sdlxliff_file_stat_record(output_path)
        except OSError:
            return False, None

    try:
        recorded_output_size = int(record["output_size"])
        recorded_output_mtime = int(record["output_mtime_ns"])
    except (KeyError, TypeError, ValueError):
        return None, None
    refreshed = None
    output_signature_matches = (
        recorded_output_size == output_stat["size"]
        and recorded_output_mtime == output_stat["mtime_ns"]
    )
    if not output_signature_matches:
        try:
            if _sdlxliff_sha256_file(output_path) != expected_output_hash:
                return False, None
        except OSError:
            return False, None
        refreshed = dict(record)
        refreshed["output_size"] = output_stat["size"]
        refreshed["output_mtime_ns"] = output_stat["mtime_ns"]

    expected_source_hash = str(record.get("source_sha256") or "").lower()
    source_record = record.get("source")
    if re.fullmatch(r"[0-9a-f]{64}", expected_source_hash) and isinstance(source_record, dict):
        source_kind = source_record.get("kind")
        tracked_path = (
            source_record.get("path")
            if source_kind == "file"
            else source_record.get("container") if source_kind == "epub_member" else None
        )
        # A moved or temporarily unavailable source must not invalidate a
        # perfectly usable existing sidecar.  If it is available, however,
        # verify any changed source signature by content.
        if tracked_path:
            try:
                if isinstance(source_stat_cache, dict):
                    source_cache_key = os.path.normcase(
                        os.path.abspath(tracked_path)
                    )
                    if source_cache_key not in source_stat_cache:
                        try:
                            source_stat_cache[source_cache_key] = (
                                _sdlxliff_file_stat_record(tracked_path)
                            )
                        except OSError:
                            source_stat_cache[source_cache_key] = None
                    source_stat = source_stat_cache.get(source_cache_key)
                    if not isinstance(source_stat, dict):
                        return True, refreshed
                else:
                    source_stat = _sdlxliff_file_stat_record(tracked_path)
                recorded_source_size = int(source_record["size"])
                recorded_source_mtime = int(source_record["mtime_ns"])
                source_signature_matches = (
                    recorded_source_size == source_stat["size"]
                    and recorded_source_mtime == source_stat["mtime_ns"]
                )
                if not source_signature_matches:
                    source_payload = _sdlxliff_source_record_payload(source_record)
                    if source_payload is None:
                        return False, None
                    current_source_hash = _sdlxliff_sha256_bytes(source_payload.encode("utf-8"))
                    if current_source_hash != expected_source_hash:
                        return False, None
                    refreshed = dict(refreshed or record)
                    refreshed_source = dict(source_record)
                    refreshed_source.update(source_stat)
                    refreshed["source"] = refreshed_source
            except (KeyError, OSError, TypeError, ValueError):
                pass

    return True, refreshed


def _manual_editing_output_filename(entry, current_output, retain_source_extension):
    """Return the HTML filename a manual sidecar must save for this naming mode."""
    entry = entry if isinstance(entry, dict) else {}
    source_name = ""
    for key in (
        "original_name", "original_basename", "original_filename",
        "filename", "href",
    ):
        candidate = os.path.basename(
            str(entry.get(key) or "").replace("\\", "/")
        )
        if candidate and candidate.lower().endswith(_PROGRESS_READER_HTML_EXTENSIONS):
            source_name = candidate
            break

    current_name = os.path.basename(
        str(current_output or entry.get("output_file") or "").replace("\\", "/")
    )
    if not source_name:
        source_name = current_name
    if source_name.lower().startswith("response_"):
        source_name = source_name[len("response_"):]
    if not source_name:
        return current_name

    if retain_source_extension:
        return source_name

    source_stem, _source_ext = os.path.splitext(source_name)
    return f"response_{source_stem or source_name}.html"


def _sdlxliff_machine_translation_output_name(path_or_name):
    name = os.path.basename(str(path_or_name or "").replace("\\", "/"))
    suffix = ".sdlxliff"
    if name.lower().endswith(suffix):
        name = name[:-len(suffix)]
    return name


def _sdlxliff_machine_translation_path(output_dir, output_or_sidecar):
    output_name = _sdlxliff_machine_translation_output_name(output_or_sidecar)
    if not output_name:
        return None
    base_dir = str(output_dir or "")
    if not base_dir:
        sidecar_path = str(output_or_sidecar or "")
        sidecar_dir = os.path.dirname(sidecar_path)
        if os.path.basename(sidecar_dir).lower() == "sdlxliff":
            base_dir = os.path.dirname(sidecar_dir)
    if not base_dir:
        return None
    safe_name = re.sub(r'[^A-Za-z0-9._ -]+', '_', output_name).strip(" .")
    if not safe_name:
        safe_name = hashlib.sha256(output_name.encode("utf-8", errors="replace")).hexdigest()
    return os.path.join(base_dir, "SDLXLIFF", _MACHINE_TRANSLATION_DIR, f"{safe_name}.json")

def _get_app_dir() -> str:
    """Return the application's base directory (Windows-safe)."""
    if platform.system() == 'Windows':
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


class SdlxliffAutogenMixin:
    """Sidecar auto-generation for a Progress Manager owner (RetranslationMixin on desktop)."""

    _SDLXLIFF_AUTOGEN_STATUSES = {
        "completed",
        "qa_failed",
        "completed_empty",
        "completed_image_only",
    }
    _SDLXLIFF_MANUAL_UNTRANSLATED_STATUSES = {
        "not_translated",
        "not translated",
        "not_completed",
        "pending",
    }

    @staticmethod
    def _sdlxliff_autogen_output_path(output_dir, output_file):
        if not output_dir or not output_file:
            return None
        normalized = str(output_file).replace("\\", "/")
        path = normalized if os.path.isabs(normalized) else os.path.join(output_dir, normalized)
        return os.path.normpath(path)

    @staticmethod
    def _output_dir_has_sdlxliff_sidecars(output_dir):
        sidecar_dir = os.path.join(output_dir or "", "SDLXLIFF")
        try:
            if not os.path.isdir(sidecar_dir):
                return False
            with os.scandir(sidecar_dir) as entries:
                return any(entry.is_file() and entry.name.lower().endswith(".sdlxliff") for entry in entries)
        except Exception:
            return False

    @classmethod
    def _output_dir_has_sdlxliff_generatable_html(cls, output_dir, progress_data=None, output_files=None):
        if not output_dir or not os.path.isdir(output_dir):
            return False
        chapters = progress_data.get("chapters") if isinstance(progress_data, dict) else None
        if not isinstance(chapters, dict):
            chapters = progress_data if isinstance(progress_data, dict) else {}
        if not isinstance(chapters, dict):
            return False

        requested = None
        if output_files:
            requested = {
                os.path.basename(str(name or "").replace("\\", "/")).lower()
                for name in output_files
                if name
            }
            requested.discard("")

        for entry in chapters.values():
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", "") or "").lower()
            if status not in cls._SDLXLIFF_AUTOGEN_STATUSES:
                continue
            output_file = entry.get("output_file")
            output_name = os.path.basename(str(output_file or "").replace("\\", "/")).lower()
            if not output_name.endswith((".html", ".htm", ".xhtml")):
                continue
            if requested is not None and output_name not in requested:
                continue
            output_path = cls._sdlxliff_autogen_output_path(output_dir, output_file)
            if output_path and os.path.isfile(output_path):
                return True
        return False

    @staticmethod
    def _sdlxliff_autogen_source_candidates(entry, output_file=None, progress_key=None):
        entry = entry if isinstance(entry, dict) else {}
        raw_candidates = [
            entry.get("original_basename"),
            entry.get("original_filename"),
            entry.get("chapter_file"),
            entry.get("source_filename"),
            entry.get("filename"),
            progress_key,
        ]
        if output_file:
            output_name = os.path.basename(str(output_file).replace("\\", "/"))
            if output_name.lower().startswith("response_"):
                raw_candidates.append(output_name[len("response_"):])
        candidates = []
        seen = set()
        for candidate in raw_candidates:
            if not candidate:
                continue
            text = str(candidate).replace("\\", "/")
            variants = [text, os.path.basename(text)]
            stem, ext = os.path.splitext(text)
            if not ext and stem:
                variants.extend([f"{text}.xhtml", f"{text}.html", f"{text}.htm"])
                base = os.path.basename(text)
                variants.extend([f"{base}.xhtml", f"{base}.html", f"{base}.htm"])
            elif ext.lower() in (".html", ".htm", ".xhtml"):
                for html_ext in (".xhtml", ".html", ".htm"):
                    variants.append(f"{stem}{html_ext}")
                base = os.path.basename(text)
                base_stem, _base_ext = os.path.splitext(base)
                if base_stem:
                    for html_ext in (".xhtml", ".html", ".htm"):
                        variants.append(f"{base_stem}{html_ext}")
            for variant in variants:
                if variant and variant not in seen:
                    seen.add(variant)
                    candidates.append(variant)
        return candidates

    @staticmethod
    def _sdlxliff_autogen_decode(data):
        for encoding in ("utf-8", "utf-8-sig", "cp949", "latin-1"):
            try:
                return data.decode(encoding)
            except Exception:
                continue
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _sdlxliff_is_extracted_epub_dir(path):
        if not path or not os.path.isdir(path):
            return False
        direct_candidates = [
            os.path.join(path, "content.opf"),
            os.path.join(path, "OEBPS", "content.opf"),
            os.path.join(path, "EPUB", "content.opf"),
            os.path.join(path, "META-INF", "container.xml"),
        ]
        if any(os.path.isfile(candidate) for candidate in direct_candidates):
            return True
        try:
            for _root_dir, _dirs, files in os.walk(path):
                if any(str(fname).lower().endswith(".opf") for fname in files):
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def _sdlxliff_autogen_read_source_from_directory(root_dir, candidate_names, candidate_basenames):
        if not root_dir or not os.path.isdir(root_dir):
            return None, None
        try:
            root_abs = os.path.abspath(root_dir)
            for dirpath, _dirs, files in os.walk(root_abs):
                for fname in files:
                    rel = os.path.relpath(os.path.join(dirpath, fname), root_abs).replace("\\", "/")
                    normalized = rel.lower().strip("/")
                    basename = os.path.basename(normalized)
                    if normalized not in candidate_names and basename not in candidate_basenames:
                        continue
                    path = os.path.join(dirpath, fname)
                    try:
                        with open(path, "rb") as f:
                            return SdlxliffAutogenMixin._sdlxliff_autogen_decode(f.read()), path
                    except Exception:
                        continue
        except Exception:
            return None, None
        return None, None

    def _sdlxliff_current_input_file_candidates(self):
        candidates = []

        def _add_many(values):
            if not values:
                return
            if isinstance(values, (str, bytes, os.PathLike)):
                values = [values]
            for value in values:
                if not value:
                    continue
                try:
                    candidates.append(str(value))
                except Exception:
                    continue

        _add_many(getattr(self, "selected_files", None))
        try:
            entry = getattr(self, "entry_epub", None)
            if entry is not None and hasattr(entry, "text"):
                _add_many(entry.text())
        except Exception:
            pass
        try:
            cfg = getattr(self, "config", None)
            if isinstance(cfg, dict):
                _add_many(cfg.get("selected_files"))
                _add_many(cfg.get("last_input_files"))
        except Exception:
            pass

        resolved = []
        seen = set()
        for candidate in candidates:
            path = os.path.normpath(candidate)
            try:
                norm = os.path.normcase(os.path.abspath(path))
            except Exception:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            if os.path.exists(path):
                resolved.append(path)
        return resolved

    def _sdlxliff_exact_input_epub_candidates(self, output_dir):
        output_name = os.path.basename(os.path.normpath(str(output_dir or "")))
        output_key = os.path.normcase(output_name)
        if not output_key:
            return []
        matches = []
        for path in self._sdlxliff_current_input_file_candidates():
            if os.path.isfile(path) and str(path).lower().endswith(".epub"):
                candidate_key = os.path.splitext(os.path.basename(path))[0]
            elif self._sdlxliff_is_extracted_epub_dir(path):
                candidate_key = os.path.basename(os.path.normpath(path))
            else:
                continue
            if os.path.normcase(candidate_key) == output_key:
                matches.append(path)
        return matches

    def _sdlxliff_valid_current_input_epub_candidates(self):
        candidates = []
        seen = set()
        for path in self._sdlxliff_current_input_file_candidates():
            resolved = self._sdlxliff_valid_epub_path("", path)
            if not resolved:
                continue
            try:
                norm = os.path.normcase(os.path.abspath(resolved))
            except Exception:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            candidates.append(resolved)
        return candidates

    @staticmethod
    def _sdlxliff_valid_epub_path(output_dir, path):
        if not path:
            return None
        candidate = path if os.path.isabs(str(path)) else os.path.join(output_dir or "", str(path))
        candidate = os.path.normpath(candidate)
        if os.path.isfile(candidate) and candidate.lower().endswith(".epub"):
            return candidate
        if SdlxliffAutogenMixin._sdlxliff_is_extracted_epub_dir(candidate):
            return candidate
        return None

    def _sdlxliff_preferred_input_epub(self, output_dir, file_path=None):
        direct = self._sdlxliff_valid_epub_path(output_dir, file_path)
        if direct:
            return direct
        exact = self._sdlxliff_exact_input_epub_candidates(output_dir)
        if exact:
            return exact[0]
        current = self._sdlxliff_valid_current_input_epub_candidates()
        return current[0] if len(current) == 1 else None

    def _sdlxliff_update_source_epub_ref(self, output_dir, epub_path):
        epub_path = self._sdlxliff_valid_epub_path(output_dir, epub_path)
        if not output_dir or not epub_path:
            return False
        source_ref = os.path.join(output_dir, "source_epub.txt")
        try:
            current = ""
            if os.path.isfile(source_ref):
                with open(source_ref, "r", encoding="utf-8", errors="ignore") as f:
                    current = f.read().strip()
            current_path = self._sdlxliff_valid_epub_path(output_dir, current)
            if current_path and os.path.normcase(os.path.abspath(current_path)) == os.path.normcase(os.path.abspath(epub_path)):
                return False
            with open(source_ref, "w", encoding="utf-8") as f:
                f.write(os.path.abspath(epub_path))
            return True
        except Exception:
            return False

    def _sdlxliff_autogen_epub_candidates(self, output_dir, file_path=None):
        candidates = []
        preferred = self._sdlxliff_preferred_input_epub(output_dir, file_path)
        if preferred:
            candidates.append(preferred)
            self._sdlxliff_update_source_epub_ref(output_dir, preferred)
        source_ref = os.path.join(output_dir or "", "source_epub.txt")
        try:
            if os.path.isfile(source_ref):
                with open(source_ref, "r", encoding="utf-8", errors="ignore") as f:
                    ref = f.read().strip()
                if ref:
                    candidates.append(ref)
        except Exception:
            pass
        candidates.extend(self._sdlxliff_exact_input_epub_candidates(output_dir))
        candidates.extend(self._sdlxliff_valid_current_input_epub_candidates())
        try:
            for fname in os.listdir(output_dir or ""):
                if str(fname).lower().endswith(".epub"):
                    candidates.append(os.path.join(output_dir, fname))
                else:
                    path = os.path.join(output_dir, fname)
                    if self._sdlxliff_is_extracted_epub_dir(path):
                        candidates.append(path)
        except Exception:
            pass
        resolved = []
        seen = set()
        for candidate in candidates:
            path = self._sdlxliff_valid_epub_path(output_dir, candidate)
            if not path:
                continue
            norm = os.path.normcase(os.path.abspath(path))
            if norm in seen:
                continue
            seen.add(norm)
            resolved.append(path)
        return resolved

    @staticmethod
    def _sdlxliff_add_spine_position(positions, name, position):
        normalized = str(name or "").replace("\\", "/").strip("/").casefold()
        if not normalized:
            return
        basename = os.path.basename(normalized)
        logical = _normalize_progress_match_name(basename).casefold()
        for key in (normalized, basename, os.path.splitext(normalized)[0], os.path.splitext(basename)[0], logical):
            if key:
                positions.setdefault(key, int(position))

    def _sdlxliff_source_spine_positions(self, output_dir, file_path=None):
        """Read authoritative HTML order from the selected EPUB's content.opf."""
        positions = {}
        candidates = self._sdlxliff_autogen_epub_candidates(output_dir, file_path)
        for source_path in candidates:
            if os.path.isfile(source_path) and source_path.lower().endswith(".epub"):
                try:
                    chapters = read_epub_spine_chapters(
                        source_path,
                        translate_special=True,
                        include_text=False,
                    )
                except Exception:
                    chapters = []
                for fallback, chapter in enumerate(chapters):
                    try:
                        position = int(chapter.get("spine_number", fallback + 1)) - 1
                    except (TypeError, ValueError):
                        position = fallback
                    self._sdlxliff_add_spine_position(
                        positions,
                        chapter.get("member_path") or chapter.get("filename"),
                        position,
                    )
                    self._sdlxliff_add_spine_position(
                        positions,
                        chapter.get("filename"),
                        position,
                    )
                if positions:
                    break
                continue

            if not os.path.isdir(source_path):
                continue
            opf_path = find_workspace_opf_path(source_path)
            if not opf_path:
                continue
            try:
                root = ET.parse(opf_path).getroot()
            except Exception:
                continue
            id_to_href = {}
            for element in root.iter():
                if str(element.tag).rsplit("}", 1)[-1] != "item":
                    continue
                item_id = element.attrib.get("id")
                href = unquote(element.attrib.get("href") or "")
                if item_id and href:
                    id_to_href[item_id] = href
            position = 0
            for element in root.iter():
                if str(element.tag).rsplit("}", 1)[-1] != "itemref":
                    continue
                href = id_to_href.get(element.attrib.get("idref"))
                if not href:
                    continue
                self._sdlxliff_add_spine_position(positions, href, position)
                position += 1
            if positions:
                break
        self._sdlxliff_cached_source_spine_positions = dict(positions)
        return positions

    @staticmethod
    def _sdlxliff_spine_position_for_entry(entry, positions):
        entry = entry if isinstance(entry, dict) else {}
        for key in (
            "original_filename", "href", "original_basename", "filename", "output_file",
        ):
            candidate = str(entry.get(key) or "").replace("\\", "/").strip("/").casefold()
            if not candidate:
                continue
            basename = os.path.basename(candidate)
            logical = _normalize_progress_match_name(basename).casefold()
            for lookup in (candidate, basename, os.path.splitext(candidate)[0], os.path.splitext(basename)[0], logical):
                if lookup in positions:
                    return positions[lookup]
        return None

    def _sdlxliff_autogen_read_source_html(self, output_dir, entry, output_file=None, progress_key=None, file_path=None):
        candidates = self._sdlxliff_autogen_source_candidates(entry, output_file, progress_key)
        candidate_names = {str(c).replace("\\", "/").lower().strip("/") for c in candidates if c}
        candidate_basenames = {os.path.basename(str(c).replace("\\", "/")).lower() for c in candidates if c}
        candidate_names.discard("")
        candidate_basenames.discard("")
        if candidate_names or candidate_basenames:
            for epub_path in self._sdlxliff_autogen_epub_candidates(output_dir, file_path):
                if os.path.isdir(epub_path):
                    source_text, source_path = self._sdlxliff_autogen_read_source_from_directory(
                        epub_path,
                        candidate_names,
                        candidate_basenames,
                    )
                    if source_text:
                        return source_text, source_path
                    continue
                try:
                    with zipfile.ZipFile(epub_path, "r") as zf:
                        for name in zf.namelist():
                            normalized = str(name).replace("\\", "/").lower().strip("/")
                            if normalized in candidate_names or os.path.basename(normalized) in candidate_basenames:
                                return self._sdlxliff_autogen_decode(zf.read(name)), f"{epub_path}!{name}"
                except Exception:
                    continue

        output_path = self._sdlxliff_autogen_output_path(output_dir, output_file)
        output_norm = os.path.normcase(os.path.abspath(output_path)) if output_path else ""
        output_base = os.path.basename(str(output_file or "").replace("\\", "/")).lower()
        for candidate in candidates:
            text = str(candidate).replace("\\", "/")
            variants = [text]
            basename = os.path.basename(text)
            if basename and basename != text:
                variants.append(basename)
            for variant in variants:
                path = variant if os.path.isabs(variant) else os.path.join(output_dir or "", variant)
                path = os.path.normpath(path)
                if output_norm and os.path.normcase(os.path.abspath(path)) == output_norm:
                    continue
                if output_base and os.path.basename(path).lower() == output_base:
                    continue
                if os.path.isfile(path):
                    try:
                        with open(path, "r", encoding="utf-8", errors="replace") as f:
                            return f.read(), path
                    except Exception:
                        continue
        return None, None

    def _sdlxliff_autogen_bulk_read_sources(self, output_dir, work_entries, file_path=None):
        """Resolve many source documents with one scan of each source EPUB."""
        candidates_by_index = {
            index: self._sdlxliff_autogen_source_candidates(entry, output_name, progress_key)
            for index, (progress_key, entry, output_name) in enumerate(work_entries)
        }
        unresolved = set(candidates_by_index)
        resolved = {}

        def _resource_maps(resources):
            exact = {}
            basename_groups = {}
            for normalized, resource in resources:
                normalized = str(normalized or "").replace("\\", "/").lower().strip("/")
                if not normalized:
                    continue
                exact.setdefault(normalized, resource)
                basename_groups.setdefault(os.path.basename(normalized), []).append(resource)
            by_basename = {
                name: matches[0]
                for name, matches in basename_groups.items()
                if len(matches) == 1
            }
            return exact, by_basename

        def _matching_resource(index, exact, by_basename):
            for candidate in candidates_by_index.get(index, ()):
                normalized = str(candidate or "").replace("\\", "/").lower().strip("/")
                if not normalized:
                    continue
                resource = exact.get(normalized)
                if resource is None:
                    resource = by_basename.get(os.path.basename(normalized))
                if resource is not None:
                    return resource
            return None

        for epub_path in self._sdlxliff_autogen_epub_candidates(output_dir, file_path):
            if not unresolved:
                break
            if os.path.isdir(epub_path):
                resources = []
                try:
                    root_abs = os.path.abspath(epub_path)
                    for dirpath, _dirs, files in os.walk(root_abs):
                        for fname in files:
                            path = os.path.join(dirpath, fname)
                            relative = os.path.relpath(path, root_abs).replace("\\", "/")
                            resources.append((relative, path))
                except Exception:
                    resources = []
                exact, by_basename = _resource_maps(resources)
                for index in list(unresolved):
                    source_path = _matching_resource(index, exact, by_basename)
                    if not source_path:
                        continue
                    try:
                        with open(source_path, "rb") as source_file:
                            source_html = self._sdlxliff_autogen_decode(source_file.read())
                    except Exception:
                        continue
                    if source_html:
                        resolved[index] = (source_html, source_path)
                        unresolved.discard(index)
                continue

            try:
                with zipfile.ZipFile(epub_path, "r") as source_zip:
                    exact, by_basename = _resource_maps(
                        (name, name) for name in source_zip.namelist()
                    )
                    for index in list(unresolved):
                        member = _matching_resource(index, exact, by_basename)
                        if not member:
                            continue
                        try:
                            source_html = self._sdlxliff_autogen_decode(source_zip.read(member))
                        except Exception:
                            continue
                        if source_html:
                            resolved[index] = (source_html, f"{epub_path}!{member}")
                            unresolved.discard(index)
            except Exception:
                continue

        # Preserve the existing loose-file fallbacks for unusual workspaces.
        for index in list(unresolved):
            progress_key, entry, output_name = work_entries[index]
            source_html, source_path = self._sdlxliff_autogen_read_source_html(
                output_dir,
                entry,
                output_file=output_name,
                progress_key=progress_key,
                file_path=file_path,
            )
            if source_html:
                resolved[index] = (source_html, source_path)
        return resolved

    @staticmethod
    def _sdlxliff_sidecar_current_for_output(
        sidecar_path,
        output_path,
        output_dir=None,
        output_name=None,
        manifest=None,
        manifest_updates=None,
        output_stat=None,
        sidecar_mtime_ns=None,
        source_stat_cache=None,
    ):
        """Prefer hash freshness, falling back to the legacy mtime rule.

        The fallback is intentionally per record: a pre-manifest workspace, a
        malformed manifest, or an old sidecar missing its record behaves
        exactly as it did before hash tracking was introduced.
        """
        try:
            if not sidecar_path or not output_path:
                return False
            if sidecar_mtime_ns is None and not os.path.isfile(sidecar_path):
                return False
            if output_stat is None and not os.path.isfile(output_path):
                return False
            if output_dir is None:
                output_dir = os.path.dirname(os.path.dirname(os.path.abspath(sidecar_path)))
            if manifest is None:
                manifest = _read_sdlxliff_sidecar_manifest(output_dir)
            logical_key = _sdlxliff_logical_output_key(output_name or sidecar_path)
            entries = manifest.get("entries") if isinstance(manifest, dict) else None
            record = entries.get(logical_key) if isinstance(entries, dict) else None
            hash_current, refreshed_record = _sdlxliff_manifest_record_current(
                record,
                output_path,
                output_stat=output_stat,
                source_stat_cache=source_stat_cache,
            )
            if hash_current is not None:
                if refreshed_record is not None:
                    if isinstance(manifest_updates, dict):
                        manifest_updates[logical_key] = refreshed_record
                    else:
                        _update_sdlxliff_sidecar_manifest(
                            output_dir,
                            {logical_key: refreshed_record},
                        )
                return hash_current

            if sidecar_mtime_ns is None:
                sidecar_stat = os.stat(sidecar_path)
                sidecar_mtime = getattr(
                    sidecar_stat,
                    "st_mtime_ns",
                    int(sidecar_stat.st_mtime * 1000000000),
                )
            else:
                sidecar_mtime = int(sidecar_mtime_ns)
            if isinstance(output_stat, dict):
                output_mtime = int(output_stat.get("mtime_ns", -1))
            else:
                output_file_stat = os.stat(output_path)
                output_mtime = getattr(
                    output_file_stat,
                    "st_mtime_ns",
                    int(output_file_stat.st_mtime * 1000000000),
                )
            return sidecar_mtime >= output_mtime
        except Exception:
            return False

    def _generate_sdlxliff_sidecars_from_completed_entries(
        self,
        output_dir,
        file_path=None,
        progress_data=None,
        output_files=None,
        overwrite=True,
        progress_callback=None,
    ):
        stats = {
            "total": 0,
            "considered": 0,
            "created": 0,
            "skipped": 0,
            "missing_source": 0,
            "missing_output": 0,
            "failed": 0,
            "paths": [],
            "errors": [],
        }
        if not output_dir:
            stats["errors"].append("Output folder was not provided")
            return stats
        if not os.path.isdir(output_dir):
            stats["errors"].append(f"Output folder does not exist: {output_dir}")
            return stats

        if progress_data is None:
            progress_path = os.path.join(output_dir, "translation_progress.json")
            try:
                with open(progress_path, "r", encoding="utf-8") as f:
                    progress_data = json.load(f)
            except Exception:
                progress_data = {}

        chapters = progress_data.get("chapters") if isinstance(progress_data, dict) else None
        if not isinstance(chapters, dict):
            chapters = progress_data if isinstance(progress_data, dict) else {}
        if not isinstance(chapters, dict):
            stats["errors"].append("translation_progress.json did not contain chapter entries")
            return stats

        requested = None
        if output_files:
            requested = {
                os.path.basename(str(name).replace("\\", "/")).lower()
                for name in output_files
                if name
            }
            requested.discard("")

        work_entries = []
        seen_outputs = set()
        for progress_key, entry in chapters.items():
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", "") or "").lower()
            if status not in self._SDLXLIFF_AUTOGEN_STATUSES:
                continue
            output_file = entry.get("output_file")
            output_name = os.path.basename(str(output_file or "").replace("\\", "/"))
            if not output_name.lower().endswith((".html", ".htm", ".xhtml")):
                continue
            if requested is not None and output_name.lower() not in requested:
                continue
            output_identity = _sdlxliff_logical_output_key(output_name)
            if output_identity in seen_outputs:
                stats["skipped"] += 1
                continue
            seen_outputs.add(output_identity)
            work_entries.append((progress_key, entry, output_name))

        total = len(work_entries)
        stats["total"] = total
        if total == 0:
            stats["errors"].append("No completed HTML entries are eligible for SDLXLIFF generation")

        def _notify(stage, index=0, output_name="", path="", message="", error=""):
            if not callable(progress_callback):
                return
            try:
                progress_callback({
                    "stage": stage,
                    "index": index,
                    "total": total,
                    "output_name": output_name,
                    "path": path,
                    "message": message,
                    "error": error,
                    "stats": dict(stats),
                })
            except Exception:
                pass

        _notify("start", total and 1 or 0, message=f"Preparing {total} SDLXLIFF sidecar(s)")
        existing_sidecars = _existing_sdlxliff_sidecars_by_logical_output(output_dir)
        sidecar_manifest = _read_sdlxliff_sidecar_manifest(output_dir) or {}
        sidecar_manifest_updates = {}
        sidecar_manifest_chunk_flush_failed = False
        old_output_sdlxliff = os.environ.get("OUTPUT_SDLXLIFF")
        os.environ["OUTPUT_SDLXLIFF"] = "1"
        try:
            for entry_index, (progress_key, entry, output_name) in enumerate(work_entries, 1):
                output_file = entry.get("output_file")
                stats["considered"] += 1
                _notify("checking", entry_index, output_name)

                logical_key = _sdlxliff_logical_output_key(output_name)
                sidecar_path = existing_sidecars.get(logical_key) or os.path.join(
                    output_dir, "SDLXLIFF", f"{output_name}.sdlxliff"
                )
                writer_output_name = (
                    SdlxliffReviewCoreMixin._sidecar_output_name(sidecar_path)
                    if os.path.isfile(sidecar_path)
                    else output_name
                )
                output_path = self._sdlxliff_autogen_output_path(output_dir, output_file)
                if not output_path or not os.path.isfile(output_path):
                    stats["missing_output"] += 1
                    _notify("missing_output", entry_index, output_name, message=f"Output HTML not found: {output_file}")
                    continue
                if os.path.isfile(sidecar_path):
                    if not overwrite:
                        stats["skipped"] += 1
                        stats["paths"].append(sidecar_path)
                        _notify("skipped", entry_index, output_name, path=sidecar_path)
                        continue
                    if requested is None and self._sdlxliff_sidecar_current_for_output(
                        sidecar_path,
                        output_path,
                        output_dir=output_dir,
                        output_name=output_name,
                        manifest=sidecar_manifest,
                        manifest_updates=sidecar_manifest_updates,
                    ):
                        stats["skipped"] += 1
                        _notify("skipped", entry_index, output_name, path=sidecar_path)
                        continue
                try:
                    with open(output_path, "r", encoding="utf-8", errors="replace") as f:
                        target_html = f.read()
                except Exception:
                    stats["missing_output"] += 1
                    message = f"Could not read output HTML: {output_file}"
                    stats["errors"].append(message)
                    _notify("missing_output", entry_index, output_name, message=message)
                    continue

                source_html, source_path = self._sdlxliff_autogen_read_source_html(
                    output_dir,
                    entry,
                    output_file=output_file,
                    progress_key=progress_key,
                    file_path=file_path,
                )
                if not source_html:
                    stats["missing_source"] += 1
                    _notify("missing_source", entry_index, output_name, message=f"Source HTML not found for {output_name}")
                    continue

                chapter = dict(entry)
                if not chapter.get("original_basename"):
                    chapter["original_basename"] = os.path.basename(str(source_path or output_name).split("!", 1)[-1])
                writer_error = ""
                try:
                    result_path = _write_html_sdlxliff_sidecar(
                        output_dir,
                        writer_output_name,
                        chapter,
                        source_html,
                        target_html,
                        raise_errors=True,
                        record_freshness=False,
                        preserve_review_metadata=True,
                    )
                except Exception as exc:
                    result_path = None
                    writer_error = f"{type(exc).__name__}: {exc}"
                    stats["errors"].append(f"{output_name}: {writer_error}")
                if result_path:
                    existing_sidecars[logical_key] = result_path
                    try:
                        sidecar_manifest_updates[logical_key] = (
                            _sdlxliff_manifest_freshness_record(
                                output_name,
                                result_path,
                                output_path,
                                source_html,
                                source_path,
                            )
                        )
                    except Exception:
                        # The sidecar remains valid.  A failed record update
                        # simply preserves the legacy mtime fallback.
                        pass
                    # The shared writer normally persists one live-translation
                    # record immediately.  Bulk reviewer generation disables
                    # that per-file write and flushes records in chunks instead
                    # of rewriting a growing JSON document thousands of times.
                    if (
                        len(sidecar_manifest_updates) >= 100
                        and not sidecar_manifest_chunk_flush_failed
                    ):
                        if _update_sdlxliff_sidecar_manifest(
                            output_dir,
                            sidecar_manifest_updates,
                        ):
                            sidecar_manifest_updates.clear()
                        else:
                            sidecar_manifest_chunk_flush_failed = True
                    stats["created"] += 1
                    stats["paths"].append(result_path)
                    _notify("created", entry_index, output_name, path=result_path)
                else:
                    stats["failed"] += 1
                    _notify("failed", entry_index, output_name, error=writer_error)
        finally:
            if sidecar_manifest_updates:
                _update_sdlxliff_sidecar_manifest(
                    output_dir,
                    sidecar_manifest_updates,
                )
            if old_output_sdlxliff is None:
                os.environ.pop("OUTPUT_SDLXLIFF", None)
            else:
                os.environ["OUTPUT_SDLXLIFF"] = old_output_sdlxliff

        _notify("finished", total, path=stats["paths"][-1] if stats["paths"] else "")
        return stats

    def _generate_sdlxliff_sidecars_from_untranslated_entries(
        self,
        output_dir,
        untranslated_entries,
        file_path=None,
        progress_callback=None,
    ):
        """Create source-only manual sidecars from Progress Manager rows."""
        stats = {
            "total": 0,
            "considered": 0,
            "created": 0,
            "skipped": 0,
            "missing_source": 0,
            "missing_output": 0,
            "failed": 0,
            "paths": [],
            "errors": [],
        }
        if not output_dir or not os.path.isdir(output_dir):
            stats["errors"].append(f"Output folder does not exist: {output_dir}")
            return stats

        entries = untranslated_entries
        if isinstance(entries, dict):
            entries = entries.get("chapters", entries)
            entries = list(entries.items()) if isinstance(entries, dict) else list(entries or [])
        else:
            entries = list(enumerate(entries or []))

        work_entries = []
        seen_outputs = set()
        retain_source_extension = (
            str(os.getenv("RETAIN_SOURCE_EXTENSION", "0")).strip().lower()
            in {"1", "true", "yes", "on"}
            or bool(
                (getattr(self, "config", {}) or {}).get(
                    "retain_source_extension",
                    False,
                )
            )
        )
        for entry_key, entry in entries:
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", "") or "").strip().lower()
            if status not in self._SDLXLIFF_MANUAL_UNTRANSLATED_STATUSES:
                continue
            manual_entry = dict(entry)
            if not manual_entry.get("original_basename"):
                manual_entry["original_basename"] = entry.get("filename")
            if not manual_entry.get("original_filename"):
                manual_entry["original_filename"] = entry.get("href")
            output_file = entry.get("output_file")
            output_name = _manual_editing_output_filename(
                manual_entry,
                output_file,
                retain_source_extension,
            )
            if not output_name.lower().endswith((".html", ".htm", ".xhtml")):
                continue
            output_key = _sdlxliff_logical_output_key(output_name)
            if output_key in seen_outputs:
                stats["skipped"] += 1
                continue
            seen_outputs.add(output_key)
            work_entries.append((str(entry_key), manual_entry, output_name))

        # Progress JSON insertion order is not EPUB reading order. Resolve and
        # sort against the source content.opf before emitting any sidecars so
        # the live-streamed viewer is correct from its first row onward.
        spine_positions = self._sdlxliff_source_spine_positions(
            output_dir,
            file_path=file_path,
        )
        ordered_work_entries = []
        for original_index, work_entry in enumerate(work_entries):
            entry = work_entry[1]
            opf_position = self._sdlxliff_spine_position_for_entry(
                entry,
                spine_positions,
            )
            entry["_manual_opf_position"] = opf_position
            ordered_work_entries.append((opf_position, original_index, work_entry))
        ordered_work_entries.sort(
            key=lambda item: (
                item[0] is None,
                item[0] if item[0] is not None else item[1],
                item[1],
            )
        )
        work_entries = [item[2] for item in ordered_work_entries]

        total = len(work_entries)
        stats["total"] = total

        def _notify(stage, index=0, output_name="", path="", message="", error="", opf_position=None):
            if not callable(progress_callback):
                return
            try:
                progress_callback({
                    "stage": stage,
                    "index": index,
                    "total": total,
                    "output_name": output_name,
                    "path": path,
                    "message": message,
                    "error": error,
                    "opf_position": opf_position,
                    "stats": dict(stats),
                })
            except Exception:
                pass

        _notify("start", message=f"Preparing {total} untranslated SDLXLIFF sidecar(s)")
        existing_sidecars = _existing_sdlxliff_sidecars_by_logical_output(output_dir)
        pending_entries = []
        for entry_index, work_entry in enumerate(work_entries, 1):
            _entry_key, _entry, output_name = work_entry
            output_path = self._sdlxliff_autogen_output_path(output_dir, output_name)
            sidecar_path = existing_sidecars.get(
                _sdlxliff_logical_output_key(output_name)
            ) or os.path.join(output_dir, "SDLXLIFF", f"{output_name}.sdlxliff")
            if (output_path and os.path.isfile(output_path)) or os.path.isfile(sidecar_path):
                stats["considered"] += 1
                stats["skipped"] += 1
                if (
                    os.path.isfile(sidecar_path)
                    and _blank_manual_untranslated_sdlxliff_target(sidecar_path)
                ):
                    stats["paths"].append(sidecar_path)
                _notify("checking", entry_index, output_name)
                _notify("skipped", entry_index, output_name, path=sidecar_path)
                continue
            pending_entries.append((entry_index, work_entry))

        source_pairs = self._sdlxliff_autogen_bulk_read_sources(
            output_dir,
            [work_entry for _entry_index, work_entry in pending_entries],
            file_path=file_path,
        )
        old_output_sdlxliff = os.environ.get("OUTPUT_SDLXLIFF")
        os.environ["OUTPUT_SDLXLIFF"] = "1"
        try:
            for pending_index, (entry_index, (entry_key, entry, output_name)) in enumerate(pending_entries):
                stats["considered"] += 1
                _notify("checking", entry_index, output_name)
                source_html, source_path = source_pairs.get(pending_index, (None, None))
                if not source_html:
                    stats["missing_source"] += 1
                    _notify(
                        "missing_source",
                        entry_index,
                        output_name,
                        message=f"Source HTML not found for {output_name}",
                    )
                    continue

                chapter = dict(entry)
                if not chapter.get("original_basename"):
                    chapter["original_basename"] = os.path.basename(str(source_path or output_name).split("!", 1)[-1])
                writer_error = ""
                try:
                    result_path = _write_html_sdlxliff_sidecar(
                        output_dir,
                        output_name,
                        chapter,
                        source_html,
                        source_html,
                        raise_errors=True,
                        manual_untranslated=True,
                    )
                except Exception as exc:
                    result_path = None
                    writer_error = f"{type(exc).__name__}: {exc}"
                    stats["errors"].append(f"{output_name}: {writer_error}")
                if result_path:
                    existing_sidecars[_sdlxliff_logical_output_key(output_name)] = result_path
                    stats["created"] += 1
                    stats["paths"].append(result_path)
                    _notify(
                        "created",
                        entry_index,
                        output_name,
                        path=result_path,
                        opf_position=entry.get("_manual_opf_position"),
                    )
                else:
                    stats["failed"] += 1
                    _notify("failed", entry_index, output_name, error=writer_error)
        finally:
            if old_output_sdlxliff is None:
                os.environ.pop("OUTPUT_SDLXLIFF", None)
            else:
                os.environ["OUTPUT_SDLXLIFF"] = old_output_sdlxliff

        _notify("finished", total, path=stats["paths"][-1] if stats["paths"] else "")
        return stats


class SdlxliffReviewCoreMixin:
    """The widget-free half of SDLXLIFFReviewDialog (see the module docstring)."""

    # List containers are excluded so their child text is not counted twice.
    # Each <li> is a standalone unit, while a void <hr> is represented as a
    # paragraph-like ***** separator so it consumes the correct row/ordinal.
    TEXT_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "hr")
    THEME = {
        "bg": "#1e1e1e",
        "panel": "#2d2d2d",
        "panel_alt": "#242424",
        "border": "#4a5568",
        "accent": "#5a9fd4",
        "info": "#17a2b8",
        "success": "#28a745",
        "warning": "#d39e00",
        "purple": "#b967ff",
        "danger": "#dc3545",
        "text": "#ffffff",
        "muted": "#94a3b8",
    }
    REVIEW_ROW_MIN_HEIGHT = 96
    REVIEW_ROW_MAX_HEIGHT = 1600
    REVIEW_TAG_LABEL_WIDTH = 96
    REVIEW_TAG_LABEL_MAX_FONT_PT = 11.0
    REVIEW_TAG_LABEL_MIN_FONT_PT = 4.0
    # Minimum height of the editable output entry in the two-column layout
    # (2nd column). 72px ≈ two text lines (2*22 + 28 padding) so short
    # entries get a comfortable click/edit target instead of the old
    # one-line 50px sliver.
    REVIEW_TARGET_EDIT_MIN_HEIGHT = 72
    REVIEW_PRELOAD_RADIUS = 2
    REVIEW_PRELOAD_BATCH_SIZE = 8
    REVIEW_PRELOAD_IDLE_MS = 350
    REVIEW_PRELOAD_STEP_MS = 90
    REVIEW_MAX_CACHED_PAGES = 7
    REVIEW_SYNC_RENDER_ROW_LIMIT = 80
    TRANSLATE_TOOLTIPS_BUTTON_TEXT = "🌐 Generate Machine Translation Preview"
    FLAG_ACCURACY_BUTTON_TEXT = "🟣 Flag Inaccurate"
    TWO_COLUMN_LAYOUT_BUTTON_TEXT = "Compact"
    NOTEPAD_LAYOUT_BUTTON_TEXT = "Notepad"
    TWO_COLUMN_LAYOUT_CONFIG_KEY = "sdlxliff_two_column_layout"
    LEGACY_ONE_COLUMN_LAYOUT_CONFIG_KEY = "sdlxliff_one_column_layout"
    LEGACY_ONE_ROW_LAYOUT_CONFIG_KEY = "sdlxliff_one_row_layout"
    MACHINE_TRANSLATION_PROVIDER_CONFIG_KEY = "sdlxliff_machine_translation_provider"
    MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY = "sdlxliff_machine_translation_deepl_api_key"
    MACHINE_TRANSLATION_BING_API_KEY_CONFIG_KEY = "sdlxliff_machine_translation_bing_api_key"
    MACHINE_TRANSLATION_BING_REGION_CONFIG_KEY = "sdlxliff_machine_translation_bing_region"
    MACHINE_TRANSLATION_YANDEX_API_KEY_CONFIG_KEY = "sdlxliff_machine_translation_yandex_api_key"
    MACHINE_TRANSLATION_YANDEX_FOLDER_ID_CONFIG_KEY = "sdlxliff_machine_translation_yandex_folder_id"
    MACHINE_TRANSLATION_API_KEY_CONFIG_KEYS = frozenset({
        MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY,
        MACHINE_TRANSLATION_BING_API_KEY_CONFIG_KEY,
        MACHINE_TRANSLATION_YANDEX_API_KEY_CONFIG_KEY,
    })
    MACHINE_TRANSLATION_PROVIDER_LABELS = {
        "auto": "Auto",
        "google": "Google",
        "deepl": "DeepL",
        "bing": "Bing",
        "argos": "Argos Translate",
        "yandex": "Yandex",
    }
    MACHINE_TRANSLATION_THRESHOLD_CONFIG_KEY = "sdlxliff_machine_translation_inaccuracy_threshold"
    MACHINE_TRANSLATION_INACCURACY_THRESHOLD = 150.0
    MACHINE_TRANSLATION_SHORT_TEXT_MAX_TOKENS = 1
    MACHINE_TRANSLATION_SHORT_TEXT_MAX_CHARS = 24
    MANUAL_GREEN_OVERRIDES_FILE = "review_status_overrides.json"
    MANUAL_GREEN_STATUSES = frozenset({"red", "yellow"})
    MACHINE_TRANSLATION_CONTENT_STOPWORDS = frozenset("""
        a an and are as at be been being but by for from had has have he her hers him his i if in into is it
        its me my of on or our ours she so some that the their theirs them then there they this to was were
        while who whom whose will with would you your yours
    """.split())
    MACHINE_TRANSLATION_PENDING_TEXT = "⏳ Generating machine translation preview..."
    MANUAL_REFRESH_BUTTON_TEXT = "↻ Refresh"
    MANUAL_EDITING_CONFIG_KEY = "retranslation_manual_editing"
    _SDLXLIFF_AUTOGEN_STATUSES = {
        "completed",
        "qa_failed",
        "completed_empty",
        "completed_image_only",
    }

    _GT_LANG_CODES = {
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "russian": "ru",
        "arabic": "ar",
        "hindi": "hi",
        "chinese": "zh-CN",
        "chinese (simplified)": "zh-CN",
        "simplified chinese": "zh-CN",
        "chinese (traditional)": "zh-TW",
        "traditional chinese": "zh-TW",
        "japanese": "ja",
        "korean": "ko",
        "turkish": "tr",
        "vietnamese": "vi",
    }

    def _build_review_refresh_scan_result(
        self,
        force=False,
        validate=False,
        current_path=None,
        last_review_signature=None,
        last_mt_signature=None,
        last_autogen_signature=None,
    ):
        result = {
            "force": bool(force),
            "validate": bool(validate),
            "current_path": current_path,
            "review_signature": last_review_signature,
            "machine_translation_signature": last_mt_signature,
            "autogen_signature": last_autogen_signature,
            "sidecar_changed": False,
            "sidecar_path_set_changed": False,
            "settings_changed": False,
            "changed_sidecar_paths": [],
            "machine_translation_changed": False,
            "autogen_changed": False,
            "sidecars_generated": False,
            "image_assets": None,
            "stats": None,
            "error": "",
        }
        try:
            try:
                image_assets_output = os.path.normcase(
                    os.path.abspath(self.output_dir or "")
                )
            except Exception:
                image_assets_output = str(self.output_dir or "")
            if (
                last_review_signature is None
                or force
                or image_assets_output
                != getattr(self, "_review_image_assets_output", "")
            ):
                result["image_assets"] = self._ensure_review_image_assets()
                if (
                    isinstance(result["image_assets"], dict)
                    and result["image_assets"].get("ready")
                ):
                    self._review_image_assets_output = image_assets_output
            review_signature = self._current_review_signature()
            mt_signature = self._current_machine_translation_signature()
            autogen_signature = self._current_review_autogen_signature()
            sidecar_changed = review_signature != last_review_signature
            mt_changed = mt_signature != last_mt_signature
            autogen_changed = autogen_signature != last_autogen_signature
            stats = None
            generated = False
            if force or validate or sidecar_changed or autogen_changed:
                stats = self._regenerate_review_sidecars_for_refresh_scan(
                    force=force,
                    previous_signature=last_autogen_signature,
                    current_signature=autogen_signature,
                    validate=validate,
                )
                generated = bool(stats and (stats.get("created") or stats.get("paths")))
                if force or generated:
                    review_signature = self._current_review_signature()
                    mt_signature = self._current_machine_translation_signature()
                    autogen_signature = self._current_review_autogen_signature()
                    sidecar_changed = True
                    mt_changed = mt_signature != last_mt_signature
                    autogen_changed = autogen_signature != last_autogen_signature
            result.update({
                "review_signature": review_signature,
                "machine_translation_signature": mt_signature,
                "autogen_signature": autogen_signature,
                "sidecar_changed": sidecar_changed,
                "sidecar_path_set_changed": self._review_signature_path_set_changed(last_review_signature, review_signature),
                "settings_changed": self._review_signature_settings(review_signature) != self._review_signature_settings(last_review_signature),
                "changed_sidecar_paths": self._changed_review_signature_paths(last_review_signature, review_signature, stats),
                "machine_translation_changed": mt_changed,
                "autogen_changed": autogen_changed,
                "sidecars_generated": generated,
                "stats": stats,
            })
        except Exception as exc:
            result["error"] = str(exc)
        return result

    @staticmethod
    def _review_signature_path_map(signature):
        path_map = {}
        for entry in signature or ():
            try:
                if not entry or entry[0] != "sdlxliff" or len(entry) < 4:
                    continue
                path_map[str(entry[1])] = tuple(entry[2:])
            except Exception:
                continue
        return path_map

    @staticmethod
    def _review_signature_settings(signature):
        """Non-file portion of the review signature (e.g. the dedupe toggle).

        Changes here don't map to any sidecar path, so the incremental reload
        would short-circuit; callers treat a settings change as a full rebuild.
        """
        settings = []
        for entry in signature or ():
            try:
                if entry and entry[0] == "review_settings":
                    settings.append(tuple(entry))
            except Exception:
                continue
        return tuple(sorted(settings))

    @classmethod
    def _review_signature_path_set_changed(cls, old_signature, new_signature):
        old_paths = set(cls._review_signature_path_map(old_signature))
        new_paths = set(cls._review_signature_path_map(new_signature))
        return old_paths != new_paths

    @classmethod
    def _changed_review_signature_paths(cls, old_signature, new_signature, stats=None):
        old_map = cls._review_signature_path_map(old_signature)
        new_map = cls._review_signature_path_map(new_signature)
        changed = {
            path for path in (set(old_map) | set(new_map))
            if old_map.get(path) != new_map.get(path)
        }
        if isinstance(stats, dict):
            for path in stats.get("paths") or []:
                try:
                    changed.add(os.path.normcase(os.path.abspath(path)))
                except Exception:
                    continue
        return sorted(changed)

    @staticmethod
    def _merge_sdlxliff_generation_stats(*results):
        results = [result for result in results if isinstance(result, dict)]
        if not results:
            return None
        merged = {
            "total": 0,
            "considered": 0,
            "created": 0,
            "skipped": 0,
            "missing_source": 0,
            "missing_output": 0,
            "failed": 0,
            "paths": [],
            "errors": [],
        }
        for result in results:
            for key in (
                "total", "considered", "created", "skipped",
                "missing_source", "missing_output", "failed",
            ):
                merged[key] += int(result.get(key) or 0)
            merged["paths"].extend(result.get("paths") or [])
            merged["errors"].extend(result.get("errors") or [])
        merged["paths"] = list(dict.fromkeys(merged["paths"]))
        return merged

    def _regenerate_manual_review_sidecars_for_refresh_scan(self):
        if not bool(
            isinstance(getattr(self, "_config", None), dict)
            and self._config.get(self.MANUAL_EDITING_CONFIG_KEY, False)
        ):
            return None
        entries = list(getattr(self, "_sdlxliff_autogen_manual_entries", None) or [])
        if not entries:
            return None
        owner = getattr(self, "_sdlxliff_autogen_owner", None)
        generator = getattr(owner, "_generate_sdlxliff_sidecars_from_untranslated_entries", None)
        if not callable(generator):
            return None

        # Auto-refresh runs every few seconds. Passing the full Progress
        # Manager list here made every ordinary editor save replay a visible
        # "Skipped SDLXLIFF n/total" sweep across the entire book. The manual
        # generator never overwrites an existing output or sidecar, so filter
        # those entries before invoking it instead of paying to rediscover the
        # same fact one item at a time.
        retain_source_extension = (
            str(os.getenv("RETAIN_SOURCE_EXTENSION", "0")).strip().lower()
            in {"1", "true", "yes", "on"}
            or bool((getattr(owner, "config", {}) or {}).get(
                "retain_source_extension",
                False,
            ))
        )
        existing_sidecars = _existing_sdlxliff_sidecars_by_logical_output(
            self.output_dir
        )
        missing_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", "") or "").strip().lower()
            if status not in {
                "not_translated", "not translated", "not_completed", "pending",
            }:
                continue
            output_name = _manual_editing_output_filename(
                entry,
                entry.get("output_file"),
                retain_source_extension,
            )
            if not str(output_name or "").lower().endswith(
                _PROGRESS_READER_HTML_EXTENSIONS
            ):
                continue
            output_path = os.path.normpath(os.path.join(
                self.output_dir or "",
                str(output_name).replace("/", os.sep),
            ))
            if os.path.isfile(output_path):
                continue
            if _sdlxliff_logical_output_key(output_name) in existing_sidecars:
                continue
            missing_entries.append(entry)
        if not missing_entries:
            return None

        file_path = getattr(self, "_sdlxliff_autogen_file_path", None)
        try:
            if 0 <= self._book_index < len(self._book_entries):
                file_path = file_path or self._book_entries[self._book_index].get("epub_path") or None
        except Exception:
            pass
        return generator(
            self.output_dir,
            missing_entries,
            file_path=file_path,
            progress_callback=self._emit_review_generation_progress,
        )

    def _regenerate_review_sidecars_for_refresh_scan(self, force=False, previous_signature=None, current_signature=None, validate=False):
        manual_stats = self._regenerate_manual_review_sidecars_for_refresh_scan()
        signature = current_signature or ()
        if not self._review_autogen_has_output_html(signature):
            return manual_stats
        missing_outputs = self._missing_review_sidecar_outputs(self.output_dir, signature)
        stale_outputs = self._stale_review_sidecar_outputs(self.output_dir, signature)
        invalid_outputs = []
        initial_scan = previous_signature is None and not force and not validate
        if initial_scan:
            # First scan after opening the dialog: skip the parse-validation of
            # every sidecar - it reads and parses all of them (twice each) and
            # held the "Checking SDLXLIFF sidecars..." screen for seconds.
            # Missing sidecars are still generated here; a deferred validate
            # pass runs in the background after the initial load and repairs
            # any invalid sidecars seamlessly.
            invalid_outputs = []
        else:
            invalid_outputs = self._invalid_review_sidecar_outputs(self.output_dir)
        if not force and signature == previous_signature and not invalid_outputs and not missing_outputs and not stale_outputs:
            return manual_stats
        if not force and previous_signature is None and not invalid_outputs and not missing_outputs and not stale_outputs:
            return manual_stats

        owner = getattr(self, "_sdlxliff_autogen_owner", None)
        generator = getattr(owner, "_generate_sdlxliff_sidecars_from_completed_entries", None)
        if not callable(generator):
            return {
                "total": 0,
                "considered": 0,
                "created": 0,
                "skipped": 0,
                "missing_source": 0,
                "missing_output": 0,
                "failed": 1,
                "paths": [],
                "errors": ["SDLXLIFF auto-generation helper is not available in this build"],
            }

        file_path = getattr(self, "_sdlxliff_autogen_file_path", None)
        try:
            if 0 <= self._book_index < len(self._book_entries):
                file_path = file_path or self._book_entries[self._book_index].get("epub_path") or None
        except Exception:
            file_path = file_path or None

        requested_output_files = list(getattr(self, "_sdlxliff_autogen_output_files", None) or [])
        output_files = requested_output_files or None
        if force:
            output_files = self._review_autogen_output_names(signature) or None
        elif not force:
            changed_outputs = [] if previous_signature is None else self._changed_review_autogen_outputs(previous_signature, signature)
            generated_outputs = sorted(set(
                (changed_outputs or [])
                + (invalid_outputs or [])
                + (missing_outputs or [])
                + (stale_outputs or [])
            ))
            if requested_output_files and previous_signature is None:
                requested_names = {
                    os.path.basename(str(name).replace("\\", "/")).lower()
                    for name in requested_output_files
                    if name
                }
                generated_outputs = [
                    name for name in generated_outputs
                    if os.path.basename(str(name).replace("\\", "/")).lower() in requested_names
                ] or requested_output_files
            output_files = generated_outputs or None
            if not output_files:
                return manual_stats

        completed_stats = generator(
            self.output_dir,
            file_path=file_path,
            progress_data=None,
            output_files=output_files,
            overwrite=True,
            progress_callback=self._emit_review_generation_progress,
        )
        return self._merge_sdlxliff_generation_stats(manual_stats, completed_stats)

    def _emit_review_log_message(self, message):
        """Send worker-safe viewer status to the Translator GUI log."""
        message = str(message or "").strip()
        if not message:
            return
        candidates = (
            getattr(self, "_sdlxliff_autogen_owner", None),
            getattr(self, "_context_parent", None),
        )
        seen = set()
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            append_log = getattr(candidate, "append_log", None)
            if not callable(append_log):
                continue
            try:
                append_log(message, source_thread=threading.current_thread())
                return
            except TypeError:
                try:
                    append_log(message)
                    return
                except Exception:
                    pass
            except Exception:
                pass

    def _queue_generated_sidecar_stream_piece(self, path, progress_index=0, opf_position=None):
        try:
            if not getattr(self, "_generation_streaming_active", False):
                return False
            if not path:
                return False
            pending = getattr(self, "_generation_stream_pending_pieces", None)
            if not isinstance(pending, list):
                pending = []
                self._generation_stream_pending_pieces = pending
            pending.append((str(path), int(progress_index or 0), opf_position))
            self._schedule_generation_stream_flush(delay_ms=0 if len(pending) <= 1 else 16)
            return True
        except Exception:
            return False

    def _schedule_generation_stream_flush(self, delay_ms=16):
        timer = getattr(self, "_generation_stream_flush_timer", None)
        if timer is None:
            return
        try:
            if not timer.isActive():
                timer.start(max(0, int(delay_ms or 0)))
        except RuntimeError:
            pass
        except Exception:
            pass

    def _review_row_rendered_or_rendering(self, row):
        try:
            row = int(row)
        except Exception:
            return False
        try:
            if row in self._piece_render_complete and self._piece_pages.get(row) is not None:
                return True
            if getattr(self, "_active_render_row", None) == row and getattr(self, "_active_render_page", None) is not None:
                return True
            if getattr(self, "_preload_render_row", None) == row and self._piece_pages.get(row) is not None:
                return True
        except Exception:
            return False
        return False

    def _trace_review_perf(self, label, started=None, force=False, **details):
        """SDLXLIFF viewer performance logging is intentionally disabled."""
        return None

    def _review_piece_worker_count(self, total):
        """Return the SDLXLIFF viewer worker count from Other Settings."""
        try:
            total = int(total)
        except Exception:
            total = 0
        if total <= 1:
            return 1

        parent = getattr(self, "_context_parent", None)
        config = getattr(parent, "config", None)
        if not isinstance(config, dict):
            config = self._config if isinstance(getattr(self, "_config", None), dict) else {}

        enabled = True
        try:
            if hasattr(parent, "enable_parallel_extraction_var"):
                enabled = bool(getattr(parent, "enable_parallel_extraction_var"))
            elif "enable_parallel_extraction" in config:
                enabled = bool(config.get("enable_parallel_extraction", True))
        except Exception:
            enabled = True
        if not enabled:
            return 1

        raw_workers = None
        try:
            if hasattr(parent, "extraction_workers_var"):
                raw_workers = getattr(parent, "extraction_workers_var")
        except Exception:
            raw_workers = None
        if raw_workers is None and "extraction_workers" in config:
            raw_workers = config.get("extraction_workers")
        if raw_workers is None:
            raw_workers = os.environ.get("EXTRACTION_WORKERS", "2")

        try:
            workers = int(str(raw_workers).strip())
        except (TypeError, ValueError):
            workers = 1
        return min(total, max(1, workers))

    def _review_generation_summary(self, stats):
        if not isinstance(stats, dict):
            return ""
        total = int(stats.get("total") or 0)
        considered = int(stats.get("considered") or 0)
        created = int(stats.get("created") or 0)
        skipped = int(stats.get("skipped") or 0)
        missing_source = int(stats.get("missing_source") or 0)
        missing_output = int(stats.get("missing_output") or 0)
        failed = int(stats.get("failed") or 0)
        errors = stats.get("errors") if isinstance(stats.get("errors"), list) else []
        details = []
        if missing_source:
            details.append(f"missing source {missing_source}")
        if missing_output:
            details.append(f"missing output {missing_output}")
        if failed:
            details.append(f"failed {failed}")
        if skipped:
            details.append(f"skipped {skipped}")
        if errors:
            details.append(str(errors[0]))
        if total or considered or created:
            denominator = total or considered or created
            summary = f"Generated {created}/{denominator} SDLXLIFF sidecar(s)"
            if details:
                summary += f": {', '.join(details)}"
            return summary
        if errors:
            return f"No SDLXLIFF sidecars generated: {errors[0]}"
        return ""

    def _manual_green_overrides_path_for_output_dir(self, output_dir):
        return os.path.join(output_dir or "", "SDLXLIFF", self.MANUAL_GREEN_OVERRIDES_FILE)

    def _current_review_signature(self):
        signature = []
        # Fold the "remove duplicate H1-H6+P pairs" toggle into the signature so
        # flipping it counts as a change: the next refresh (auto or F5) rebuilds
        # the rows with the new setting instead of needing a program restart.
        signature.append(("review_settings", "remove_dup_h1p", 1 if self._review_remove_duplicate_h1_p_enabled() else 0, 0))
        for output_dir in self._review_output_dirs():
            override_path = self._manual_green_overrides_path_for_output_dir(output_dir)
            signature.append(("review_settings", "manual_green_overrides") + self._review_file_signature(override_path))
            for path in self._sdlxliff_sidecar_paths_for_output_dir(output_dir):
                try:
                    stat = os.stat(path)
                    signature.append(("sdlxliff", os.path.normcase(os.path.abspath(path)), stat.st_size, getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1000000000))))
                except Exception:
                    signature.append(("sdlxliff", os.path.normcase(os.path.abspath(path)), -1, -1))
        return tuple(sorted(signature))

    def _review_output_dirs(self):
        dirs = []
        if self._book_entries:
            dirs.extend(entry.get("output_dir") for entry in self._book_entries if entry.get("output_dir"))
        elif self.output_dir:
            dirs.append(self.output_dir)
        seen = set()
        output_dirs = []
        for output_dir in dirs:
            try:
                norm_dir = os.path.normcase(os.path.abspath(output_dir))
            except Exception:
                continue
            if norm_dir in seen:
                continue
            seen.add(norm_dir)
            output_dirs.append(output_dir)
        return output_dirs

    def _current_machine_translation_signature(self):
        signature = []
        for output_dir in self._review_output_dirs():
            mt_dir = os.path.join(output_dir, "SDLXLIFF", _MACHINE_TRANSLATION_DIR)
            try:
                mt_dir_norm = os.path.normcase(os.path.abspath(mt_dir))
            except Exception:
                mt_dir_norm = str(mt_dir or "")
            signature.append(("machine_translation_dir",) + self._review_file_signature(mt_dir))
            if os.path.isdir(mt_dir):
                try:
                    for name in sorted(os.listdir(mt_dir)):
                        if not str(name).lower().endswith(".json"):
                            continue
                        mt_path = os.path.join(mt_dir, name)
                        if os.path.isfile(mt_path):
                            signature.append(("machine_translation",) + self._review_file_signature(mt_path))
                except Exception:
                    signature.append(("machine_translation_scan_failed", mt_dir_norm, -1, -1))
        return tuple(sorted(signature))

    @staticmethod
    def _review_file_signature(path):
        try:
            norm = os.path.normcase(os.path.abspath(path))
        except Exception:
            norm = str(path or "")
        try:
            stat = os.stat(path)
            mtime = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1000000000))
            return (norm, stat.st_size, mtime)
        except Exception:
            return (norm, -1, -1)

    def _manual_green_overrides_path_for_piece(self, piece):
        try:
            sidecar_path = str((piece or {}).get("path") or "")
            if sidecar_path:
                return os.path.join(os.path.dirname(os.path.abspath(sidecar_path)), self.MANUAL_GREEN_OVERRIDES_FILE)
        except Exception:
            pass
        return self._manual_green_overrides_path_for_output_dir(getattr(self, "output_dir", "") or "")

    @staticmethod
    def _manual_green_override_key(piece):
        try:
            name = os.path.basename(str((piece or {}).get("path") or ""))
            if not name:
                name = os.path.basename(str((piece or {}).get("name") or ""))
            if not name:
                name = os.path.basename(str((piece or {}).get("output_name") or ""))
            return name.lower()
        except Exception:
            return ""

    @staticmethod
    def _manual_green_piece_content_hash(piece):
        rows_payload = []
        try:
            for row in (piece or {}).get("rows") or []:
                rows_payload.append({
                    "source_tag": str(row.get("source_tag", "") or ""),
                    "source": str(row.get("source", "") or ""),
                    "target_tag": str(row.get("target_tag", "") or ""),
                    "target": str(row.get("target", "") or ""),
                })
        except Exception:
            rows_payload = []
        raw = json.dumps(rows_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    def _progress_path_for_review_piece(self, piece):
        """Return the progress file belonging to a sidecar's output folder."""
        try:
            sidecar_path = os.path.abspath(str((piece or {}).get("path") or ""))
            sidecar_dir = os.path.dirname(sidecar_path)
            if sidecar_path and os.path.basename(sidecar_dir).casefold() == "sdlxliff":
                return os.path.join(os.path.dirname(sidecar_dir), "translation_progress.json")
        except Exception:
            pass
        return os.path.join(getattr(self, "output_dir", "") or "", "translation_progress.json")

    @staticmethod
    def _review_progress_chapters(progress_data):
        if not isinstance(progress_data, dict):
            return {}
        chapters = progress_data.get("chapters")
        if isinstance(chapters, dict):
            return chapters
        # Retain support for the legacy top-level chapter-entry layout used by
        # the existing Progress Manager reader.
        return progress_data

    @staticmethod
    def _refresh_review_progress_completed_list(progress_data):
        """Keep ProgressManager's derived completed-list cache in sync."""
        if not isinstance(progress_data, dict):
            return
        chapters = SdlxliffReviewCoreMixin._review_progress_chapters(progress_data)
        completed = []
        for key, entry in list(chapters.items()):
            if not isinstance(entry, dict) or entry.get("special_type"):
                continue
            if str(entry.get("status") or "").lower() != "completed":
                continue
            output_file = entry.get("output_file")
            if not output_file:
                continue
            actual_num = entry.get("actual_num", 0)
            completed.append({
                "num": actual_num,
                "idx": 0,
                "title": entry.get("original_basename") or f"Chapter {actual_num}",
                "file": output_file,
                "key": key,
            })
        try:
            completed.sort(key=lambda item: float(item.get("num", 0)))
        except (TypeError, ValueError):
            completed.sort(key=lambda item: str(item.get("num", "")))
        progress_data["completed_list"] = completed

    @staticmethod
    def _write_review_progress_data(path, progress_data):
        if not path or not isinstance(progress_data, dict):
            return False
        tmp_path = ""
        try:
            progress_dir = os.path.dirname(path)
            if progress_dir:
                os.makedirs(progress_dir, exist_ok=True)
            tmp_path = (
                f"{path}.{os.getpid()}.{threading.get_ident()}."
                f"{time.time_ns()}.tmp"
            )
            with open(tmp_path, "w", encoding="utf-8") as progress_file:
                json.dump(progress_data, progress_file, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
            return True
        except Exception:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return False

    def _sync_cached_review_progress_data(self, progress_path, progress_data):
        """Update the live Progress Manager data passed to this review window."""
        cached = getattr(self, "_sdlxliff_autogen_progress_data", None)
        if not isinstance(cached, dict):
            return
        expected_path = os.path.join(getattr(self, "output_dir", "") or "", "translation_progress.json")
        try:
            if os.path.normcase(os.path.abspath(progress_path)) != os.path.normcase(os.path.abspath(expected_path)):
                return
            replacement = copy.deepcopy(progress_data)
            cached.clear()
            cached.update(replacement)
        except Exception:
            pass

    def _retain_source_extension_enabled(self):
        config = self._config if isinstance(getattr(self, "_config", None), dict) else {}
        return (
            str(os.getenv("RETAIN_SOURCE_EXTENSION", "0")).strip().lower()
            in {"1", "true", "yes", "on"}
            or bool(config.get("retain_source_extension", False))
        )

    def _manual_output_name_for_piece(self, piece, current_output=None):
        current_output = current_output or (
            (piece or {}).get("output_name")
            or self._sidecar_output_name((piece or {}).get("path") or (piece or {}).get("name") or "")
        )
        return _manual_editing_output_filename(
            piece,
            current_output,
            self._retain_source_extension_enabled(),
        )

    def _matching_review_progress_entries(self, progress_data, piece, *extra_names):
        """Match a sidecar to progress across response-prefix/extension renames."""
        chapters = self._review_progress_chapters(progress_data)
        progress_key = str((piece or {}).get("progress_key") or "")
        exact_names = {
            self._canonical_basename(name)
            for name in (
                (piece or {}).get("output_name"),
                self._sidecar_output_name((piece or {}).get("path") or ""),
                *extra_names,
            )
            if name
        }
        logical_names = {
            _normalize_progress_match_name(name).casefold()
            for name in (
                *exact_names,
                (piece or {}).get("original_name"),
            )
            if name and _normalize_progress_match_name(name)
        }

        matches = []
        seen = set()
        for key, entry in list(chapters.items()):
            if not isinstance(entry, dict):
                continue
            entry_output = entry.get("output_file")
            entry_original = entry.get("original_basename") or entry.get("original_filename")
            exact_match = self._canonical_basename(entry_output) in exact_names
            logical_match = any(
                _normalize_progress_match_name(candidate).casefold() in logical_names
                for candidate in (entry_output, entry_original)
                if candidate
            )
            if str(key) == progress_key or exact_match or logical_match:
                marker = id(entry)
                if marker not in seen:
                    matches.append((key, entry))
                    seen.add(marker)
        return matches

    def _mark_piece_progress_pending(self, piece, previous_output_name=None):
        """Seed Pending only when this exact manually authored output has no entry."""
        progress_path = self._progress_path_for_review_piece(piece)
        if os.path.isfile(progress_path):
            try:
                with open(progress_path, "r", encoding="utf-8") as progress_file:
                    progress_data = json.load(progress_file)
            except Exception:
                return False
        else:
            progress_data = {
                "chapters": {},
                "chapter_chunks": {},
                "completed_list": [],
                "version": "2.1",
            }
        if not isinstance(progress_data, dict):
            return False

        output_name = self._manual_output_name_for_piece(piece, previous_output_name)
        chapters = self._review_progress_chapters(progress_data)
        if not isinstance(chapters, dict):
            return False

        target_output = self._canonical_basename(output_name)
        for key, entry in list(chapters.items()):
            if not isinstance(entry, dict):
                continue
            if self._canonical_basename(entry.get("output_file")) != target_output:
                continue
            # The existing progress entry is authoritative. Auto-discovery can
            # reuse it after the HTML appears; a manual save must not rewrite
            # its status, timestamps, or completion metadata.
            if not (piece or {}).get("progress_key"):
                piece["progress_key"] = str(key)
            return True

        original_name = (
            (piece or {}).get("original_name")
            or (piece or {}).get("original_basename")
            or os.path.basename(output_name)
        )
        actual_num = (piece or {}).get("chapter_num")
        if actual_num is None:
            actual_num = self._chapter_number_from_name(
                original_name or output_name
            )
        try:
            actual_num = int(actual_num)
        except (TypeError, ValueError):
            actual_num = self._chapter_number_from_name(output_name)

        preferred_key = str((piece or {}).get("progress_key") or "").strip()
        if not preferred_key:
            preferred_key = str(actual_num) if actual_num else (
                f"manual:{target_output}"
            )
        new_key = preferred_key
        suffix = 2
        while new_key in chapters:
            new_key = f"{preferred_key}:manual:{suffix}"
            suffix += 1
        chapters[new_key] = {
            "actual_num": actual_num,
            "status": "pending",
            "output_file": output_name,
            "original_basename": os.path.basename(str(original_name)),
            "manual_editing_pending": True,
            "last_updated": time.time(),
        }
        piece["progress_key"] = str(new_key)
        piece["output_name"] = output_name
        piece["manual_editing_pending"] = True
        self._refresh_review_progress_completed_list(progress_data)
        if not self._write_review_progress_data(progress_path, progress_data):
            return False
        self._sync_cached_review_progress_data(progress_path, progress_data)
        return True

    def _mark_piece_progress_completed(self, piece):
        """Mark every progress row matching this piece's output file completed."""
        progress_path = self._progress_path_for_review_piece(piece)
        if not os.path.isfile(progress_path):
            return {"ok": False, "matched": 0, "error": "translation_progress.json was not found"}
        try:
            with open(progress_path, "r", encoding="utf-8") as progress_file:
                progress_data = json.load(progress_file)
        except Exception as exc:
            return {"ok": False, "matched": 0, "error": f"could not read progress: {exc}"}
        if not isinstance(progress_data, dict):
            return {"ok": False, "matched": 0, "error": "translation progress is invalid"}

        output_name = self._output_name_for_piece(piece)
        target_name = self._canonical_basename(output_name)
        matches = self._matching_review_progress_entries(
            progress_data,
            piece,
            output_name,
        )
        if not target_name or not matches:
            return {
                "ok": False,
                "matched": 0,
                "error": f"no progress entry matches {output_name}",
            }

        previous_entries = {
            str(key): copy.deepcopy(entry)
            for key, entry in matches
        }
        now = time.time()
        for _key, entry in matches:
            entry["status"] = "completed"
            entry["last_updated"] = now
            entry["manually_marked_completed"] = True
            entry.pop("manual_editing_pending", None)
            for field in (
                "qa_issues", "qa_issues_found", "qa_issue_previews",
                "qa_timestamp", "failure_reason", "error_message",
                "previous_status", "previous_progress_entry",
                "previous_status_unknown", "merged_parent_chapter",
            ):
                entry.pop(field, None)
        piece.pop("manual_editing_pending", None)
        self._refresh_review_progress_completed_list(progress_data)
        if not self._write_review_progress_data(progress_path, progress_data):
            return {"ok": False, "matched": 0, "error": "could not save translation progress"}

        piece["manual_green_progress_path"] = progress_path
        piece["manual_green_previous_progress_entries"] = previous_entries
        piece["manual_green_completion_updated_at"] = now
        self._sync_cached_review_progress_data(progress_path, progress_data)
        try:
            self._last_autogen_signature = self._current_review_autogen_signature()
        except Exception:
            pass
        return {"ok": True, "matched": len(matches), "error": ""}

    def _restore_piece_progress_before_manual_completion(self, piece, override_entry=None):
        """Restore the progress rows captured by Mark as Completed."""
        override_entry = override_entry if isinstance(override_entry, dict) else {}
        previous_entries = (piece or {}).get("manual_green_previous_progress_entries")
        if not isinstance(previous_entries, dict):
            previous_entries = override_entry.get("previous_progress_entries")
        if not isinstance(previous_entries, dict) or not previous_entries:
            # Legacy green overrides predate progress mutation and need no
            # progress restoration.
            return True

        progress_path = str(
            (piece or {}).get("manual_green_progress_path")
            or override_entry.get("progress_path")
            or self._progress_path_for_review_piece(piece)
        )
        try:
            with open(progress_path, "r", encoding="utf-8") as progress_file:
                progress_data = json.load(progress_file)
        except Exception:
            return False
        if not isinstance(progress_data, dict):
            return False

        chapters = self._review_progress_chapters(progress_data)
        completion_updated_at = (
            (piece or {}).get("manual_green_completion_updated_at")
            or override_entry.get("completion_updated_at")
        )
        changed = False
        for key, previous_entry in previous_entries.items():
            current_entry = chapters.get(str(key))
            if not isinstance(current_entry, dict) or not isinstance(previous_entry, dict):
                continue
            # Do not overwrite a later real translation or another external
            # Progress Manager change made after this review action.
            if (
                str(current_entry.get("status") or "").lower() == "completed"
                and current_entry.get("manually_marked_completed") is True
                and (
                    completion_updated_at is None
                    or current_entry.get("last_updated") == completion_updated_at
                )
            ):
                chapters[str(key)] = copy.deepcopy(previous_entry)
                changed = True
        if changed:
            self._refresh_review_progress_completed_list(progress_data)
            if not self._write_review_progress_data(progress_path, progress_data):
                return False
            self._sync_cached_review_progress_data(progress_path, progress_data)
            try:
                self._last_autogen_signature = self._current_review_autogen_signature()
            except Exception:
                pass
        return True

    @staticmethod
    def _manual_green_empty_override_data():
        return {"version": 1, "entries": {}}

    def _read_manual_green_override_data(self, path):
        if not path or not os.path.isfile(path):
            return self._manual_green_empty_override_data()
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                return self._manual_green_empty_override_data()
            entries = loaded.get("entries")
            if not isinstance(entries, dict):
                entries = loaded.get("manual_green")
            if not isinstance(entries, dict):
                entries = loaded.get("overrides")
            if not isinstance(entries, dict):
                entries = {}
            return {"version": int(loaded.get("version") or 1), "entries": entries}
        except Exception:
            return self._manual_green_empty_override_data()

    def _write_manual_green_override_data(self, path, data):
        if not path:
            return False
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            entries = data.get("entries") if isinstance(data, dict) else {}
            if not isinstance(entries, dict):
                entries = {}
            payload = {
                "version": 1,
                "entries": entries,
            }
            tmp_path = f"{path}.{os.getpid()}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
            return True
        except Exception:
            try:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return False

    def _manual_green_override_entry_for_piece(self, piece):
        path = self._manual_green_overrides_path_for_piece(piece)
        key = self._manual_green_override_key(piece)
        if not path or not key:
            return None
        data = self._read_manual_green_override_data(path)
        entries = data.get("entries") if isinstance(data, dict) else {}
        entry = entries.get(key) if isinstance(entries, dict) else None
        if not isinstance(entry, dict):
            return None
        content_hash = str(entry.get("content_hash") or "")
        if not content_hash or content_hash != self._manual_green_piece_content_hash(piece):
            return None
        return entry

    def _persist_piece_manual_green_override(self, piece):
        path = self._manual_green_overrides_path_for_piece(piece)
        key = self._manual_green_override_key(piece)
        if not path or not key:
            return False
        data = self._read_manual_green_override_data(path)
        entries = data.get("entries") if isinstance(data, dict) else {}
        if not isinstance(entries, dict):
            entries = {}
        entries[key] = {
            "status": "green",
            "reason": str(piece.get("manual_green_reason") or "manually marked completed"),
            "content_hash": self._manual_green_piece_content_hash(piece),
            "sidecar": os.path.basename(str(piece.get("path") or piece.get("name") or key)),
            "output_name": self._output_name_for_piece(piece),
            "updated_at": time.time(),
        }
        previous_entries = piece.get("manual_green_previous_progress_entries")
        if isinstance(previous_entries, dict) and previous_entries:
            entries[key]["progress_path"] = str(piece.get("manual_green_progress_path") or "")
            entries[key]["previous_progress_entries"] = copy.deepcopy(previous_entries)
            entries[key]["completion_updated_at"] = piece.get("manual_green_completion_updated_at")
        data["entries"] = entries
        saved = self._write_manual_green_override_data(path, data)
        if saved:
            try:
                self._last_review_signature = self._current_review_signature()
            except Exception:
                pass
        return saved

    def _remove_piece_manual_green_override(self, piece):
        path = self._manual_green_overrides_path_for_piece(piece)
        key = self._manual_green_override_key(piece)
        if not path or not key:
            return False
        data = self._read_manual_green_override_data(path)
        entries = data.get("entries") if isinstance(data, dict) else {}
        if not isinstance(entries, dict) or key not in entries:
            return False
        entries.pop(key, None)
        data["entries"] = entries
        saved = self._write_manual_green_override_data(path, data)
        if saved:
            try:
                self._last_review_signature = self._current_review_signature()
            except Exception:
                pass
        return saved

    def _piece_needs_manual_green_override(self, piece):
        if not isinstance(piece, dict) or piece.get("manual_green_override"):
            return False
        if piece.get("manual_editing") and not piece.get("manual_untranslated"):
            return True
        if piece.get("mismatch") or int(piece.get("yellow_count") or 0) > 0:
            return True
        try:
            return any((row.get("status") in self.MANUAL_GREEN_STATUSES) for row in (piece.get("rows") or []))
        except Exception:
            return False

    def _recompute_piece_row_statuses(self, piece):
        rows = (piece or {}).get("rows") or []
        for row_data in rows:
            try:
                row_data.pop("_top_skew_promoted", None)
                row_data.pop("_machine_accuracy_promoted", None)
                row_data.pop("_machine_accuracy_previous_status", None)
                row_data.pop("_machine_accuracy_previous_reason", None)
                source_missing = bool(row_data.get(
                    "source_missing", not row_data.get("source_tag")
                ))
                target_missing = bool(row_data.get(
                    "target_missing", not row_data.get("target_tag")
                ))
                status, reason = self._row_status(
                    row_data.get("source", ""),
                    row_data.get("target", ""),
                    source_missing=source_missing,
                    target_missing=target_missing,
                )
                if row_data.get("source_tag") and row_data.get("target_tag") and row_data.get("source_tag") != row_data.get("target_tag"):
                    status, reason = self._tag_mismatch_status(row_data.get("source_tag"), row_data.get("target_tag"))
                row_data["status"] = status
                row_data["reason"] = reason
            except Exception:
                continue

    def _apply_manual_green_override_to_piece(self, piece, reason=None, content_hash=None):
        if not isinstance(piece, dict):
            return False
        piece["manual_green_override"] = True
        piece["manual_green_reason"] = str(reason or "manually marked completed")
        piece["manual_green_content_hash"] = str(content_hash or self._manual_green_piece_content_hash(piece))
        self._refresh_piece_summary(piece)
        return True

    def _apply_persisted_manual_green_override(self, piece):
        entry = self._manual_green_override_entry_for_piece(piece)
        if not isinstance(entry, dict):
            return False
        previous_entries = entry.get("previous_progress_entries")
        if isinstance(previous_entries, dict) and previous_entries:
            piece["manual_green_previous_progress_entries"] = copy.deepcopy(previous_entries)
            piece["manual_green_progress_path"] = str(
                entry.get("progress_path") or self._progress_path_for_review_piece(piece)
            )
            piece["manual_green_completion_updated_at"] = entry.get("completion_updated_at")
        return self._apply_manual_green_override_to_piece(
            piece,
            reason=entry.get("reason") or "manually marked completed",
            content_hash=entry.get("content_hash"),
        )

    def _clear_piece_manual_green_override(self, piece, persist=False):
        if not isinstance(piece, dict):
            return False
        had_override = bool(piece.get("manual_green_override"))
        if not had_override:
            return False
        override_entry = self._manual_green_override_entry_for_piece(piece) if persist else None
        if persist and not self._restore_piece_progress_before_manual_completion(piece, override_entry):
            return False
        if persist:
            override_path = self._manual_green_overrides_path_for_piece(piece)
            override_key = self._manual_green_override_key(piece)
            override_data = self._read_manual_green_override_data(override_path)
            override_entries = override_data.get("entries") if isinstance(override_data, dict) else {}
            if (
                isinstance(override_entries, dict)
                and override_key in override_entries
                and not self._remove_piece_manual_green_override(piece)
            ):
                return False
        piece.pop("manual_green_override", None)
        piece.pop("manual_green_reason", None)
        piece.pop("manual_green_content_hash", None)
        piece.pop("manual_green_progress_path", None)
        piece.pop("manual_green_previous_progress_entries", None)
        piece.pop("manual_green_completion_updated_at", None)
        return had_override

    def _current_review_autogen_signature(self):
        output_dir = self.output_dir or ""
        signature = []
        progress_path = os.path.join(output_dir, "translation_progress.json")
        source_ref_path = os.path.join(output_dir, "source_epub.txt")
        signature.append(("progress",) + self._review_file_signature(progress_path))
        signature.append(("source_epub_ref",) + self._review_file_signature(source_ref_path))

        epub_paths = []
        try:
            if os.path.isfile(source_ref_path):
                with open(source_ref_path, "r", encoding="utf-8", errors="ignore") as f:
                    ref = f.read().strip()
                if ref:
                    epub_paths.append(ref if os.path.isabs(ref) else os.path.join(output_dir, ref))
        except Exception:
            pass
        try:
            if 0 <= self._book_index < len(self._book_entries):
                epub_path = self._book_entries[self._book_index].get("epub_path")
                if epub_path:
                    epub_paths.append(epub_path)
        except Exception:
            pass
        try:
            owner = getattr(self, "_sdlxliff_autogen_owner", None)
            exact_candidates = getattr(owner, "_sdlxliff_exact_input_epub_candidates", None)
            if callable(exact_candidates):
                epub_paths.extend(exact_candidates(output_dir))
        except Exception:
            pass
        seen_epubs = set()
        for epub_path in epub_paths:
            try:
                norm = os.path.normcase(os.path.abspath(epub_path))
            except Exception:
                continue
            if norm in seen_epubs:
                continue
            seen_epubs.add(norm)
            signature.append(("source_epub",) + self._review_file_signature(epub_path))

        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
        except Exception:
            progress_data = {}
        chapters = progress_data.get("chapters") if isinstance(progress_data, dict) else None
        if not isinstance(chapters, dict):
            chapters = progress_data if isinstance(progress_data, dict) else {}
        if isinstance(chapters, dict):
            for progress_key, entry in chapters.items():
                if not isinstance(entry, dict):
                    continue
                status = str(entry.get("status", "") or "").lower()
                if status not in self._SDLXLIFF_AUTOGEN_STATUSES:
                    continue
                output_file = entry.get("output_file")
                output_name = os.path.basename(str(output_file or "").replace("\\", "/"))
                if not output_name.lower().endswith((".html", ".htm", ".xhtml")):
                    continue
                normalized = str(output_file).replace("\\", "/")
                output_path = normalized if os.path.isabs(normalized) else os.path.join(output_dir, normalized)
                signature.append((
                    "output_html",
                    str(progress_key),
                    output_name.lower(),
                    status,
                    str(entry.get("original_basename") or ""),
                    str(entry.get("original_filename") or ""),
                    str(entry.get("chapter_file") or ""),
                    str(entry.get("source_filename") or ""),
                    str(entry.get("filename") or ""),
                ) + self._review_file_signature(os.path.normpath(output_path)))

        manual_editing_enabled = bool(
            isinstance(getattr(self, "_config", None), dict)
            and self._config.get(self.MANUAL_EDITING_CONFIG_KEY, False)
        )
        if manual_editing_enabled:
            for entry_index, entry in enumerate(
                getattr(self, "_sdlxliff_autogen_manual_entries", None) or []
            ):
                if not isinstance(entry, dict):
                    continue
                status = str(entry.get("status", "") or "").strip().lower()
                if status not in {"not_translated", "not translated", "not_completed", "pending"}:
                    continue
                output_file = entry.get("output_file")
                output_name = os.path.basename(str(output_file or "").replace("\\", "/"))
                if not output_name.lower().endswith((".html", ".htm", ".xhtml")):
                    continue
                signature.append((
                    "manual_html",
                    str(entry_index),
                    output_name.lower(),
                    status,
                    str(entry.get("original_basename") or entry.get("filename") or ""),
                    str(entry.get("original_filename") or entry.get("href") or ""),
                ))
        return tuple(sorted(signature))

    def _review_source_epub_for_image_assets(self):
        """Return the exact source EPUB associated with this review workspace."""
        candidates = [getattr(self, "_sdlxliff_autogen_file_path", None)]
        try:
            if 0 <= self._book_index < len(self._book_entries):
                candidates.append(self._book_entries[self._book_index].get("epub_path"))
        except Exception:
            pass
        source_ref = os.path.join(self.output_dir or "", "source_epub.txt")
        try:
            with open(source_ref, "r", encoding="utf-8", errors="ignore") as handle:
                candidates.append(handle.read().strip())
        except OSError:
            pass
        for candidate in candidates:
            if not candidate:
                continue
            path = str(candidate)
            if not os.path.isabs(path):
                path = os.path.join(self.output_dir or "", path)
            path = os.path.normpath(path)
            if os.path.isfile(path) and path.lower().endswith(".epub"):
                return path
        return None

    def _ensure_review_image_assets(self):
        """Run only Chapter Extractor's image preparation for this workspace."""
        source_epub = self._review_source_epub_for_image_assets()
        if not source_epub or not self.output_dir:
            return None

        def _progress(message):
            message = str(message or "").strip()
            if not message:
                return
            self._emit_review_log_message(message)
            self._emit_review_generation_progress({
                "stage": "image_assets",
                "message": message,
            })

        try:
            from Chapter_Extractor import prepare_epub_image_assets

            return prepare_epub_image_assets(
                source_epub,
                self.output_dir,
                progress_callback=_progress,
            )
        except Exception as exc:
            _progress(f"❌ EPUB image asset preparation failed: {type(exc).__name__}: {exc}")
            return {
                "ready": False,
                "prepared": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    @staticmethod
    def _changed_review_autogen_outputs(previous_signature, current_signature):
        previous_signature = previous_signature or ()
        current_signature = current_signature or ()

        def _output_rows(signature):
            rows = {}
            for entry in signature:
                if not entry or entry[0] != "output_html":
                    continue
                output_name = entry[2] if len(entry) > 2 else ""
                key = (entry[1] if len(entry) > 1 else "", output_name)
                if output_name:
                    rows[key] = entry
            return rows

        previous = _output_rows(previous_signature)
        current = _output_rows(current_signature)
        changed = []
        for key, entry in current.items():
            if previous.get(key) != entry:
                changed.append(key[1])
        return sorted(set(changed))

    @staticmethod
    def _review_autogen_output_names(autogen_signature):
        output_names = []
        for entry in autogen_signature or ():
            if not entry or entry[0] != "output_html":
                continue
            output_name = entry[2] if len(entry) > 2 else ""
            if output_name:
                output_names.append(output_name)
        return sorted(set(output_names))

    @classmethod
    def _review_autogen_has_output_html(cls, autogen_signature):
        for entry in autogen_signature or ():
            if not entry or entry[0] != "output_html":
                continue
            try:
                if len(entry) > 10 and int(entry[10]) >= 0:
                    return True
            except Exception:
                continue
        return False

    def _missing_review_sidecar_outputs(self, output_dir, autogen_signature):
        expected = self._review_autogen_output_names(autogen_signature)
        if not expected:
            return []
        existing = set()
        for path in self._sdlxliff_sidecar_paths_for_output_dir(output_dir):
            output_name = self._sidecar_output_name(path)
            if output_name:
                existing.add(output_name.lower())
        return sorted(name for name in expected if name.lower() not in existing)

    def _stale_review_sidecar_outputs(self, output_dir, autogen_signature):
        sidecars = {}
        for path in self._sdlxliff_sidecar_paths_for_output_dir(output_dir):
            output_name = self._sidecar_output_name(path)
            if not output_name:
                continue
            logical_key = _sdlxliff_logical_output_key(output_name)
            if logical_key:
                try:
                    sidecar_stat = os.stat(path)
                    sidecar_mtime = int(getattr(
                        sidecar_stat,
                        "st_mtime_ns",
                        int(sidecar_stat.st_mtime * 1000000000),
                    ))
                except OSError:
                    sidecar_mtime = -1
                sidecars.setdefault(logical_key, (path, sidecar_mtime))

        # Use an empty in-memory manifest when none exists so a legacy folder
        # does not perform one filesystem lookup per chapter.
        manifest = _read_sdlxliff_sidecar_manifest(output_dir) or {}
        manifest_updates = {}
        source_stat_cache = {}
        stale = []
        for entry in autogen_signature or ():
            if not entry or entry[0] != "output_html":
                continue
            output_name = entry[2] if len(entry) > 2 else ""
            if not output_name:
                continue
            try:
                output_size = int(entry[10]) if len(entry) > 10 else -1
                output_mtime = int(entry[11]) if len(entry) > 11 else -1
            except Exception:
                continue
            logical_key = _sdlxliff_logical_output_key(output_name)
            sidecar_path, sidecar_mtime = sidecars.get(
                logical_key,
                (None, -1),
            )
            output_path = entry[9] if len(entry) > 9 else ""
            if (
                output_size >= 0
                and sidecar_path
                and sidecar_mtime >= 0
                and not SdlxliffAutogenMixin._sdlxliff_sidecar_current_for_output(
                    sidecar_path,
                    output_path,
                    output_dir=output_dir,
                    output_name=output_name,
                    manifest=manifest,
                    manifest_updates=manifest_updates,
                    output_stat={
                        "size": output_size,
                        "mtime_ns": output_mtime,
                    },
                    sidecar_mtime_ns=sidecar_mtime,
                    source_stat_cache=source_stat_cache,
                )
            ):
                stale.append(output_name)
        if manifest_updates:
            _update_sdlxliff_sidecar_manifest(output_dir, manifest_updates)
        return sorted(set(stale))

    @staticmethod
    def _review_normalized_unit_text(text):
        return " ".join(str(text or "").split())

    def _sdlxliff_sidecar_needs_source_regeneration(self, path):
        try:
            if _is_manual_untranslated_sdlxliff(path):
                return False
            source_html, target_html = self._read_sdlxliff_html_pair(path)
            source_texts = [
                self._review_normalized_unit_text(unit.get("text"))
                for unit in self._extract_text_units(source_html)
            ]
            target_texts = [
                self._review_normalized_unit_text(unit.get("text"))
                for unit in self._extract_text_units(target_html)
            ]
            source_non_empty = [text for text in source_texts if text]
            target_non_empty = [text for text in target_texts if text]
            if target_non_empty and not source_non_empty:
                return True
            if (
                source_non_empty
                and target_non_empty
                and len(source_non_empty) == len(target_non_empty)
                and source_non_empty == target_non_empty
            ):
                return True
        except Exception:
            return False
        return False

    def _invalid_review_sidecar_outputs(self, output_dir):
        invalid = []
        for path in self._sdlxliff_sidecar_paths_for_output_dir(output_dir):
            if not self._sdlxliff_sidecar_needs_source_regeneration(path):
                continue
            output_name = self._sidecar_output_name(path)
            if output_name:
                invalid.append(output_name)
        return sorted(set(invalid))

    def _invalid_review_sidecar_regen_key(self, output_files, autogen_signature):
        return (
            tuple(sorted(str(name or "").lower() for name in (output_files or []) if name)),
            autogen_signature or (),
            self._current_review_signature(),
        )

    def _maybe_regenerate_review_sidecars(self, force=False):
        previous_signature = getattr(self, "_last_autogen_signature", None)
        try:
            signature = self._current_review_autogen_signature()
        except Exception:
            signature = ()

        missing_outputs = self._missing_review_sidecar_outputs(self.output_dir, signature)
        stale_outputs = self._stale_review_sidecar_outputs(self.output_dir, signature)
        if not force and previous_signature is None:
            self._last_autogen_signature = signature
            if not missing_outputs and not stale_outputs:
                return False
            previous_signature = signature
            invalid_outputs = []
        else:
            invalid_outputs = self._invalid_review_sidecar_outputs(self.output_dir)
        invalid_regen_key = None
        if invalid_outputs:
            invalid_regen_key = self._invalid_review_sidecar_regen_key(invalid_outputs, signature)
            if not force and invalid_regen_key == getattr(self, "_last_invalid_sidecar_regen_key", None):
                invalid_outputs = []

        if not force and signature == previous_signature and not invalid_outputs and not missing_outputs and not stale_outputs:
            return False
        self._last_autogen_signature = signature

        owner = getattr(self, "_sdlxliff_autogen_owner", None)
        generator = getattr(owner, "_generate_sdlxliff_sidecars_from_completed_entries", None)
        if not callable(generator):
            return False

        file_path = None
        try:
            if 0 <= self._book_index < len(self._book_entries):
                file_path = self._book_entries[self._book_index].get("epub_path") or None
        except Exception:
            file_path = None

        output_files = None
        if force:
            output_files = self._review_autogen_output_names(signature) or None
        else:
            changed_outputs = self._changed_review_autogen_outputs(previous_signature, signature)
            output_files = sorted(set(
                (changed_outputs or [])
                + (invalid_outputs or [])
                + (missing_outputs or [])
                + (stale_outputs or [])
            )) or None

        try:
            stats = generator(
                self.output_dir,
                file_path=file_path,
                progress_data=None,
                output_files=output_files,
                overwrite=True,
            )
            if invalid_regen_key is not None and invalid_outputs:
                self._last_invalid_sidecar_regen_key = self._invalid_review_sidecar_regen_key(
                    invalid_outputs or output_files,
                    signature,
                )
            return bool(stats and (stats.get("created") or stats.get("paths")))
        except Exception:
            return False
        finally:
            try:
                self._last_autogen_signature = self._current_review_autogen_signature()
            except Exception:
                pass

    def _machine_translation_inaccuracy_threshold(self):
        try:
            value = (self._config or {}).get(
                self.MACHINE_TRANSLATION_THRESHOLD_CONFIG_KEY,
                self.MACHINE_TRANSLATION_INACCURACY_THRESHOLD,
            )
            value = float(value)
            if value <= 0:
                raise ValueError
            return max(1.0, min(1000.0, value))
        except Exception:
            return float(self.MACHINE_TRANSLATION_INACCURACY_THRESHOLD)

    def _review_two_column_layout_enabled(self):
        try:
            config = self._config or {}
            if self.TWO_COLUMN_LAYOUT_CONFIG_KEY in config:
                value = config.get(self.TWO_COLUMN_LAYOUT_CONFIG_KEY)
            elif self.LEGACY_ONE_COLUMN_LAYOUT_CONFIG_KEY in config:
                value = config.get(self.LEGACY_ONE_COLUMN_LAYOUT_CONFIG_KEY)
            elif self.LEGACY_ONE_ROW_LAYOUT_CONFIG_KEY in config:
                value = config.get(self.LEGACY_ONE_ROW_LAYOUT_CONFIG_KEY)
            else:
                return True
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        except Exception:
            return True

    @staticmethod
    def _detect_notepad_mode_support():
        """Check Lite/full package capability without importing Qt WebEngine."""
        try:
            for module_name in (
                "PySide6.QtWebEngineCore",
                "PySide6.QtWebEngineWidgets",
            ):
                if module_name in sys.modules:
                    continue
                if importlib.util.find_spec(module_name) is None:
                    return False
            return True
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def _review_notepad_mode_is_available(self):
        supported = getattr(self, "_notepad_mode_supported", None)
        if supported is None:
            supported = self._detect_notepad_mode_support()
            self._notepad_mode_supported = bool(supported)
        return bool(supported)

    def _persist_review_config_value(self, key, value):
        if key in self.MACHINE_TRANSLATION_API_KEY_CONFIG_KEYS:
            value = self._encrypt_machine_translation_api_key(value)
        if isinstance(self._config, dict):
            self._config[key] = value
        parent = getattr(self, "_context_parent", None)
        try:
            parent_config = getattr(parent, "config", None)
            if isinstance(parent_config, dict):
                parent_config[key] = value
        except Exception:
            pass
        try:
            save_config = getattr(parent, "save_config", None)
            if callable(save_config):
                save_config(show_message=False)
                return True
        except TypeError:
            try:
                save_config(getattr(parent, "config", self._config), show_message=False)
                return True
            except Exception:
                pass
        except Exception:
            pass

        config_path = None
        for attr in ("config_file_path", "config_file"):
            try:
                candidate = getattr(parent, attr, None)
            except Exception:
                candidate = None
            if candidate:
                config_path = candidate
                break
        if not config_path:
            config_path = os.path.join(_get_app_dir(), "config.json")
        try:
            config_data = {}
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    config_data = loaded
            config_data[key] = value
            os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
            tmp_path = f"{config_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, config_path)
            return True
        except Exception:
            return False

    @staticmethod
    def _decrypt_machine_translation_api_key(value):
        value = str(value or "").strip()
        if not value:
            return ""
        try:
            from api_key_encryption import get_handler
            return str(get_handler().decrypt_value(value) or "").strip()
        except Exception:
            return value

    @staticmethod
    def _encrypt_machine_translation_api_key(value):
        value = str(value or "").strip()
        if not value or value.startswith("ENC:"):
            return value
        try:
            from api_key_encryption import get_handler
            return str(get_handler().encrypt_value(value) or value)
        except Exception:
            return value

    def _machine_translation_config_value(self, key, default=""):
        value = (self._config or {}).get(key, default)
        if key in self.MACHINE_TRANSLATION_API_KEY_CONFIG_KEYS:
            return self._decrypt_machine_translation_api_key(value)
        return str(value or "").strip()

    @classmethod
    def _normalize_machine_translation_provider(cls, provider):
        value = str(provider or "auto").strip().lower().replace("_", "-").replace(" ", "-")
        aliases = {
            "": "auto",
            "machine": "auto",
            "machine-translation": "auto",
            "google-free": "google",
            "google-translate": "google",
            "google-translate-free": "google",
            "argos": "argos",
            "argos-translate": "argos",
            "argostranslate": "argos",
            "microsoft": "bing",
            "microsoft-translator": "bing",
            "azure": "bing",
            "azure-translator": "bing",
            "yandex-translate": "yandex",
        }
        value = aliases.get(value, value)
        return value if value in cls.MACHINE_TRANSLATION_PROVIDER_LABELS else "auto"

    def _machine_translation_provider(self):
        try:
            return self._normalize_machine_translation_provider(
                (self._config or {}).get(self.MACHINE_TRANSLATION_PROVIDER_CONFIG_KEY, "auto")
            )
        except Exception:
            return "auto"

    def _machine_translation_provider_label(self, provider=None):
        provider = self._normalize_machine_translation_provider(provider or self._machine_translation_provider())
        return self.MACHINE_TRANSLATION_PROVIDER_LABELS.get(provider, "Auto")

    def _machine_translation_pending_text(self):
        return f"⏳ Translating with {self._machine_translation_provider_label()}..."

    @staticmethod
    def _compact_machine_translation_error(error):
        text = str(error or "").strip()
        if not text:
            return "Machine translation preview failed"
        if "All Google Translate endpoints failed" in text:
            endpoint_lines = [
                line.strip().lstrip("•").strip()
                for line in text.splitlines()
                if "translate" in line and ": " in line
            ]
            reasons = []
            for line in endpoint_lines:
                match = re.match(r"^https?://[^\s]+:\s*(.+)$", line)
                reason = (match.group(1) if match else line.rsplit(": ", 1)[-1]).strip()
                if reason:
                    reasons.append(reason)
            count = len(endpoint_lines) or len(reasons)
            if reasons and len(set(reasons)) == 1:
                reason = reasons[0].replace(": ", " ")
                return f"Google failed: {reason} on {count} endpoints"
            if count:
                return f"Google failed on {count} endpoints; see tooltip for details"
            return "Google failed on all endpoints; see tooltip for details"
        if text.startswith("Auto fell back") and "Google endpoints failed:" in text:
            provider_match = re.match(r"^Auto fell back to (.+?) after Google endpoints failed:", text)
            provider_text = f" to {provider_match.group(1).strip()}" if provider_match else ""
            endpoints = [
                item.strip()
                for item in text.split("Google endpoints failed:", 1)[-1].split(",")
                if item.strip()
            ]
            if endpoints:
                return f"Auto fell back{provider_text} after Google failed on {len(endpoints)} endpoints"
        first_line = text.splitlines()[0].strip()
        if len(first_line) > 180:
            first_line = first_line[:177].rstrip() + "..."
        return first_line

    @classmethod
    def _row_machine_translation_preview_from_snapshot(cls, row):
        row = row if isinstance(row, dict) else {}
        if row.get("tooltip_translation_pending"):
            return str(row.get("tooltip_translation_status") or cls.MACHINE_TRANSLATION_PENDING_TEXT)
        error = str(row.get("tooltip_translation_error") or "").strip()
        if error:
            return error
        return str(row.get("tooltip_translation") or "").strip()

    @staticmethod
    def _row_machine_translation_preview_state(row):
        row = row if isinstance(row, dict) else {}
        if row.get("tooltip_translation_pending"):
            return "pending"
        if str(row.get("tooltip_translation_error") or "").strip():
            return "error"
        if str(row.get("tooltip_translation") or "").strip():
            return "translation"
        return ""

    @staticmethod
    def _machine_translation_result_note(result):
        if not isinstance(result, dict):
            return ""
        note = str(result.get("fallback_note") or "").strip()
        if note:
            return note
        endpoints = result.get("fallback_failed_endpoints")
        if isinstance(endpoints, (list, tuple)) and endpoints:
            return "Auto fell back after Google endpoints failed: " + ", ".join(str(item) for item in endpoints if item)
        return ""

    @staticmethod
    def _append_machine_translation_note(message, note):
        message = str(message or "").strip()
        note = str(note or "").strip()
        if not note:
            return message
        if not message:
            return note
        return f"{message} {note}"

    def _set_machine_translation_provider(self, provider):
        provider = self._normalize_machine_translation_provider(provider)
        if provider in {"deepl", "bing", "yandex"} and not self._prompt_machine_translation_credentials(provider):
            return
        saved = self._persist_review_config_value(self.MACHINE_TRANSLATION_PROVIDER_CONFIG_KEY, provider)
        self._update_machine_translation_button_tooltip()
        try:
            suffix = "" if saved else " (not saved to config.json)"
            self.save_status_label.setText(f"Machine translation provider: {self._machine_translation_provider_label(provider)}{suffix}")
        except Exception:
            pass

    def _machine_translation_api_options(self):
        options = {}
        deepl_key = self._machine_translation_config_value(self.MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY)
        if deepl_key:
            options["deepl"] = {"api_key": deepl_key}
        bing_key = self._machine_translation_config_value(self.MACHINE_TRANSLATION_BING_API_KEY_CONFIG_KEY)
        if bing_key:
            bing_options = {"api_key": bing_key}
            region = self._machine_translation_config_value(self.MACHINE_TRANSLATION_BING_REGION_CONFIG_KEY)
            if region:
                bing_options["region"] = region
            options["bing"] = bing_options
        yandex_key = self._machine_translation_config_value(self.MACHINE_TRANSLATION_YANDEX_API_KEY_CONFIG_KEY)
        yandex_folder_id = self._machine_translation_config_value(self.MACHINE_TRANSLATION_YANDEX_FOLDER_ID_CONFIG_KEY)
        if yandex_key or yandex_folder_id:
            options["yandex"] = {
                "api_key": yandex_key,
                "folder_id": yandex_folder_id,
            }
        return options

    def _machine_translation_translator(self, target_code, status_callback=None):
        from google_free_translate import GoogleFreeTranslateNew
        return GoogleFreeTranslateNew(
            "auto",
            target_code,
            provider=self._machine_translation_provider(),
            api_keys=self._machine_translation_api_options(),
            honor_global_stop=False,
            endpoint_status_callback=status_callback,
        )

    def _set_machine_translation_inaccuracy_threshold(self, threshold):
        try:
            value = float(threshold)
        except Exception:
            value = float(self.MACHINE_TRANSLATION_INACCURACY_THRESHOLD)
        value = max(1.0, min(1000.0, value))
        if abs(value - round(value)) < 0.05:
            value = float(round(value))
        return value, self._persist_review_config_value(self.MACHINE_TRANSLATION_THRESHOLD_CONFIG_KEY, value)

    def _reset_machine_translation_threshold(self):
        threshold, saved = self._set_machine_translation_inaccuracy_threshold(self.MACHINE_TRANSLATION_INACCURACY_THRESHOLD)
        try:
            suffix = "" if saved else " (not saved to config.json)"
            self.save_status_label.setText(f"MT inaccuracy threshold reset to {threshold:g}{suffix}")
        except Exception:
            pass

    def _candidate_epub_paths_from_context(self, parent):
        # LIVE candidates = the files currently loaded in the input field /
        # dialog. HISTORICAL candidates = paths persisted in config from
        # past sessions (last_input_files / selected_files). Only the live
        # set should drive the sidecar scan: using config history made the
        # viewer enumerate + signature-scan the output folders of every
        # book the user EVER loaded, holding "Checking SDLXLIFF
        # sidecars..." on dialogs that only concern the current inputs.
        # Config history is kept solely as a fallback for a standalone
        # viewer launched with no live selection at all.
        live_candidates = []
        historical_candidates = []

        def _add_many(bucket, values):
            if not values:
                return
            for value in values:
                try:
                    path = str(value)
                except Exception:
                    continue
                if (
                    path
                    and (
                        (path.lower().endswith(".epub") and os.path.isfile(path))
                        or SdlxliffAutogenMixin._sdlxliff_is_extracted_epub_dir(path)
                    )
                ):
                    bucket.append(path)

        widget = parent
        seen_widgets = set()
        while widget is not None and id(widget) not in seen_widgets:
            seen_widgets.add(id(widget))
            _add_many(live_candidates, getattr(widget, "_epub_files_in_dialog", None))
            _add_many(live_candidates, getattr(widget, "selected_files", None))
            try:
                cfg = getattr(widget, "config", None)
                if isinstance(cfg, dict):
                    _add_many(historical_candidates, cfg.get("last_input_files"))
                    _add_many(historical_candidates, cfg.get("selected_files"))
            except Exception:
                pass
            try:
                widget = widget.parent()
            except Exception:
                widget = None

        if isinstance(self._config, dict):
            _add_many(historical_candidates, self._config.get("last_input_files"))
            _add_many(historical_candidates, self._config.get("selected_files"))

        candidates = live_candidates if live_candidates else historical_candidates

        seen = set()
        resolved = []
        for path in candidates:
            try:
                norm = os.path.normcase(os.path.abspath(path))
            except Exception:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            resolved.append(os.path.abspath(path))
        return resolved

    def _review_output_dirs_for_epub(self, epub_path):
        base = os.path.splitext(os.path.basename(epub_path))[0]
        candidates = []

        def _add(path):
            if path:
                candidates.append(os.path.normpath(path))

        override_dir = os.environ.get("OUTPUT_DIRECTORY") or os.environ.get("OUTPUT_DIR")
        if not override_dir and isinstance(self._config, dict):
            override_dir = self._config.get("output_directory")

        if override_dir:
            _add(os.path.join(override_dir, base))
        else:
            _add(base)

        try:
            current_parent = os.path.dirname(os.path.abspath(self.output_dir))
            if current_parent:
                _add(os.path.join(current_parent, base))
        except Exception:
            pass

        try:
            input_parent = os.path.dirname(os.path.abspath(epub_path))
            _add(os.path.join(input_parent, base))
        except Exception:
            pass

        if _IS_MACOS and candidates:
            mac_candidates = []
            for path in candidates:
                if not os.path.isabs(path):
                    mac_candidates.append(os.path.join(os.path.dirname(os.path.abspath(epub_path)), path))
            candidates.extend(mac_candidates)

        seen = set()
        resolved = []
        for path in candidates:
            try:
                norm = os.path.normcase(os.path.abspath(path))
            except Exception:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            resolved.append(path)
        return resolved

    def _sdlxliff_sidecar_paths_for_output_dir(self, output_dir):
        sidecar_dir = os.path.join(output_dir or "", "SDLXLIFF")
        paths = []
        try:
            if os.path.isdir(sidecar_dir):
                for fname in os.listdir(sidecar_dir):
                    if fname.lower().endswith(".sdlxliff"):
                        paths.append(os.path.join(sidecar_dir, fname))
        except Exception:
            return []
        return sorted(paths, key=lambda path: os.path.basename(path).lower())

    @staticmethod
    def _output_dir_has_sdlxliff_sidecars(output_dir):
        sidecar_dir = os.path.join(output_dir or "", "SDLXLIFF")
        try:
            if not os.path.isdir(sidecar_dir):
                return False
            with os.scandir(sidecar_dir) as entries:
                return any(entry.is_file() and entry.name.lower().endswith(".sdlxliff") for entry in entries)
        except Exception:
            return False

    def _discover_review_books(self, parent):
        entries = []
        entries_by_dir = {}
        seen_dirs = set()
        initial_output_dir = os.path.abspath(self.output_dir) if self.output_dir else ""
        initial_current_path = os.path.abspath(self.current_path) if self.current_path else ""

        def _add_entry(output_dir, epub_path=None, current_path=None):
            if not output_dir:
                return
            if not self._output_dir_has_sdlxliff_sidecars(output_dir):
                return
            try:
                abs_dir = os.path.abspath(output_dir)
                norm = os.path.normcase(abs_dir)
            except Exception:
                return
            if norm in seen_dirs:
                existing = entries_by_dir.get(norm)
                if existing is not None:
                    if epub_path and not existing.get("epub_path"):
                        existing["epub_path"] = epub_path
                        existing["label"] = os.path.splitext(os.path.basename(os.path.normpath(epub_path)))[0] or existing.get("label") or "SDLXLIFF"
                    if current_path and os.path.isfile(current_path) and not existing.get("current_path"):
                        existing["current_path"] = current_path
                return
            seen_dirs.add(norm)
            label = os.path.splitext(os.path.basename(os.path.normpath(epub_path)))[0] if epub_path else os.path.basename(abs_dir)
            selected_path = current_path if current_path and os.path.isfile(current_path) else ""
            entry = {
                "epub_path": epub_path or "",
                "output_dir": output_dir,
                "label": label or "SDLXLIFF",
                "current_path": selected_path,
            }
            entries.append(entry)
            entries_by_dir[norm] = entry

        if initial_output_dir:
            _add_entry(initial_output_dir, current_path=initial_current_path)

        for epub_path in self._candidate_epub_paths_from_context(parent):
            for output_dir in self._review_output_dirs_for_epub(epub_path):
                current_path = ""
                try:
                    if initial_current_path:
                        sidecar_root = os.path.join(os.path.abspath(output_dir), "SDLXLIFF")
                        if os.path.commonpath([sidecar_root, initial_current_path]) == sidecar_root:
                            current_path = initial_current_path
                except Exception:
                    current_path = ""
                _add_entry(output_dir, epub_path=epub_path, current_path=current_path)

        entries.sort(key=lambda entry: str(entry.get("label", "")).lower())
        return entries

    def _initial_review_book_index(self):
        if not self._book_entries:
            return 0
        current_dir = os.path.normcase(os.path.abspath(self.output_dir)) if self.output_dir else ""
        current_path = os.path.normcase(os.path.abspath(self.current_path)) if self.current_path else ""
        for index, entry in enumerate(self._book_entries):
            try:
                output_dir = os.path.normcase(os.path.abspath(entry.get("output_dir") or ""))
                if current_dir and output_dir == current_dir:
                    return index
                if current_path:
                    sidecar_dir = os.path.normcase(os.path.abspath(os.path.join(entry.get("output_dir") or "", "SDLXLIFF")))
                    if os.path.commonpath([sidecar_dir, current_path]) == sidecar_dir:
                        return index
            except Exception:
                continue
        return 0

    @staticmethod
    def _local_name(tag):
        return str(tag).rsplit("}", 1)[-1].lower()

    @staticmethod
    def _inner_xml_or_text(element):
        if element is None:
            return ""
        if len(list(element)) == 0:
            return element.text or ""
        parts = []
        if element.text:
            parts.append(element.text)
        for child in list(element):
            parts.append(ET.tostring(child, encoding="unicode"))
            if child.tail:
                parts.append(child.tail)
        return "".join(parts)

    def _read_sdlxliff_html_pair(self, path, *, include_user_added_indexes=False):
        source_parts = []
        target_parts = []
        root = ET.parse(path).getroot()
        user_added_indexes = set()
        user_added_break_positions = {}
        for element in root.iter():
            name = self._local_name(element.tag)
            if name == "source":
                source_parts.append(self._inner_xml_or_text(element))
            elif name == "target":
                target_parts.append(self._inner_xml_or_text(element))
            elif name == "file" and include_user_added_indexes:
                try:
                    stored_indexes = json.loads(
                        element.attrib.get(USER_ADDED_TARGET_INDEXES_ATTRIBUTE, "[]")
                    )
                    user_added_indexes.update(
                        int(value)
                        for value in stored_indexes
                        if int(value) >= 0
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
                try:
                    stored_positions = json.loads(
                        element.attrib.get(USER_ADDED_BREAK_POSITIONS_ATTRIBUTE, "{}")
                    )
                    if isinstance(stored_positions, dict):
                        for raw_index, raw_positions in stored_positions.items():
                            index = int(raw_index)
                            if index < 0 or not isinstance(raw_positions, list):
                                continue
                            positions = sorted({
                                int(position)
                                for position in raw_positions
                                if int(position) >= 0
                            })
                            if positions:
                                user_added_break_positions[index] = positions
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
        html_pair = ("\n".join(source_parts), "\n".join(target_parts))
        if include_user_added_indexes:
            return (*html_pair, user_added_indexes, user_added_break_positions)
        return html_pair

    @staticmethod
    def _review_row_index_property(widget, default=-1):
        try:
            value = widget.property("sdl_row_index")
            if value is None:
                return default
            return int(value)
        except Exception:
            return default

    @classmethod
    def _review_lxml_available(cls):
        available = getattr(cls, "_review_lxml_available_flag", None)
        if available is None:
            try:
                import lxml.html  # noqa: F401
                available = True
            except Exception:
                available = False
            cls._review_lxml_available_flag = available
        return available

    @staticmethod
    def _unescape_html_document(text):
        """Unescape an HTML payload ONLY when it is stored fully escaped.

        Sidecars may store the whole document HTML-escaped (no literal
        ``<`` anywhere) — those need one unescape pass before parsing.
        But a document that already contains real markup must NOT be
        blanket-unescaped: doing so turns literal text like
        ``&lt;Prologue&gt;`` into a phantom ``<Prologue>`` tag that
        swallows the heading's text, drops the unit (449 vs 450), and
        shifts every following row's source/output + machine-translation
        pairing off by one. HTML parsers decode entities inside text
        nodes themselves, so escaped text must reach them escaped.
        """
        text = str(text or "")
        if "<" not in text and ("&lt;" in text or "&gt;" in text or "&amp;" in text):
            return html_lib.unescape(text)
        return text

    def _extract_text_units(self, html_text, include_empty=False):
        text = self._unescape_html_document(html_text)
        if self._review_lxml_available():
            try:
                return self._extract_text_units_lxml(text, include_empty=include_empty)
            except Exception:
                pass  # fall back to the BeautifulSoup implementation below
        return self._extract_text_units_bs4(text, include_empty=include_empty)

    def _extract_text_units_lxml(self, text, include_empty=False):
        """Native lxml extractor (C document tree end to end).

        Verified output-identical to the BeautifulSoup implementation on all
        real sidecars (318 files / 636 source+target documents, 0 mismatches)
        and ~7x faster. lxml also releases the GIL while parsing, so the
        background piece-loading worker barely competes with the GUI thread.
        """
        import lxml.html as lxml_html
        # lxml refuses unicode strings that carry an XML encoding declaration
        text = re.sub(r"<\?xml[^>]*\?>", "", text)
        if not text.strip():
            return []
        root = lxml_html.fromstring(text)
        text_tags = set(self.TEXT_TAGS)
        units = []
        index = 0
        for el in root.iter():
            name = el.tag.lower() if isinstance(el.tag, str) else ""
            if not name:
                continue
            if name in text_tags:
                matched = True
            elif name == "div":
                classes = (el.get("class") or "").split()
                matched = "u" in {cls.lower() for cls in classes} and not any(
                    isinstance(child.tag, str) and child.tag.lower() in text_tags
                    for child in el.iterdescendants()
                )
            else:
                matched = False
            if not matched:
                continue
            current_index = index
            index += 1
            value = (
                "*****"
                if name == "hr"
                else self._normalize_review_text(
                    " ".join(
                        part.strip()
                        for part in el.itertext()
                        if part and part.strip()
                    )
                )
            )
            if not value and not include_empty:
                continue
            units.append({
                "index": current_index,
                "tag": "p" if name in {"div", "hr"} else name,
                "text": value,
            })
        return units

    def _extract_text_units_bs4(self, text, include_empty=False):
        try:
            from bs4 import BeautifulSoup
        except Exception:
            return []
        soup = BeautifulSoup(text, "html.parser")
        units = []
        text_tags = set(self.TEXT_TAGS)

        def _is_text_unit(tag):
            name = str(getattr(tag, "name", "") or "").lower()
            if name in text_tags:
                return True
            if name != "div":
                return False
            classes = tag.get("class") or []
            if isinstance(classes, str):
                classes = classes.split()
            if "u" not in {str(cls).lower() for cls in classes}:
                return False
            return not tag.find(self.TEXT_TAGS)

        for index, tag in enumerate(soup.find_all(_is_text_unit)):
            tag_name = tag.name.lower()
            value = (
                "*****"
                if tag_name == "hr"
                else self._normalize_review_text(tag.get_text(" ", strip=True))
            )
            if not value and not include_empty:
                continue
            if tag_name in {"div", "hr"}:
                tag_name = "p"
            units.append({
                "index": index,
                "tag": tag_name,
                "text": value,
            })
        return units

    def _review_remove_duplicate_h1_p_enabled(self):
        """Mirror the translator's "Remove duplicate H1-H6+P pairs" toggle.

        Reads the same ``remove_duplicate_h1_p`` config key the header
        translation pipeline uses. The live parent config is checked first so
        toggling the setting takes effect on the next refresh without a program
        restart; the ``REMOVE_DUPLICATE_H1_P`` env var the packaged app exports
        is the final fallback.
        """
        for cfg in (
            getattr(getattr(self, "_context_parent", None), "config", None),
            getattr(self, "_config", None),
        ):
            try:
                if isinstance(cfg, dict) and "remove_duplicate_h1_p" in cfg:
                    return bool(cfg.get("remove_duplicate_h1_p"))
            except Exception:
                continue
        try:
            return str(os.environ.get("REMOVE_DUPLICATE_H1_P", "")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            return False

    @staticmethod
    def _heading_paragraph_dedupe_key(text):
        """Comparison key matching html_duplicate_cleanup._comparison_text."""
        import unicodedata
        value = unicodedata.normalize("NFKC", str(text or "").replace("\xa0", " ").strip())
        return " ".join(value.split()).casefold()

    def _dedupe_heading_paragraph_units(self, units):
        """Drop a <p> unit that duplicates an adjacent heading's text.

        This reproduces ``remove_duplicate_heading_paragraph_pairs`` at the
        extracted-unit level so the reviewer hides the same title echoes the
        translation pipeline strips when the setting is on. Each surviving
        unit keeps its original ``index`` (the position of its node in the raw
        HTML), so target write-back and machine-translation keys stay valid —
        only the duplicate is omitted from the displayed/aligned rows.
        """
        units = list(units or [])
        if len(units) < 2:
            return units
        key = self._heading_paragraph_dedupe_key
        drop = set()
        count = len(units)
        for idx, unit in enumerate(units):
            tag = str((unit or {}).get("tag", "") or "").strip().lower()
            if not re.fullmatch(r"h[1-6]", tag):
                continue
            heading_key = key(unit.get("text", ""))
            if not heading_key:
                continue
            # Prefer the following <p> (matches check_next), then the previous
            # one (check_previous); skip units already marked for removal.
            nxt = idx + 1
            if nxt < count and nxt not in drop:
                cand = units[nxt]
                if str((cand or {}).get("tag", "")).strip().lower() == "p" and key(cand.get("text", "")) == heading_key:
                    drop.add(nxt)
                    continue
            prv = idx - 1
            if prv >= 0 and prv not in drop:
                cand = units[prv]
                if str((cand or {}).get("tag", "")).strip().lower() == "p" and key(cand.get("text", "")) == heading_key:
                    drop.add(prv)
        if not drop:
            return units
        return [unit for i, unit in enumerate(units) if i not in drop]

    @staticmethod
    def _normalize_review_text(text):
        text = html_lib.unescape(str(text or ""))
        text = re.sub(r"[\u200b\u200c\u200d\ufeff\u2060]", "", text)
        text = text.replace("\xa0", " ")
        return text.strip()

    @staticmethod
    def _non_empty_text_unit_count(units):
        try:
            return sum(1 for unit in (units or []) if SdlxliffReviewCoreMixin._normalize_review_text(unit.get("text", "")))
        except Exception:
            return 0

    @staticmethod
    def _non_empty_text_units(units):
        try:
            return [
                unit for unit in (units or [])
                if SdlxliffReviewCoreMixin._normalize_review_text(unit.get("text", ""))
            ]
        except Exception:
            return []

    @staticmethod
    def _annotate_review_tag_labels(units):
        counts = Counter()
        for unit in units or []:
            tag = str((unit or {}).get("tag", "") or "").strip().lower()
            if not tag:
                continue
            # Paragraphs and list items occupy the same sequential text-block
            # stream in the review viewer. Keep the real tag in the label, but
            # share its ordinal so a p -> li conversion does not shift every
            # following paragraph number (p(10) -> li(10), then p(11)).
            counter_key = "p/li" if tag in {"p", "li"} else tag
            counts[counter_key] += 1
            ordinal = counts[counter_key]
            unit["tag_ordinal"] = ordinal
            unit["tag_label"] = tag if ordinal == 1 else f"{tag}({ordinal})"
        return units

    @staticmethod
    def _has_linguistic_letters(text):
        return any(ch.isalpha() for ch in str(text or ""))

    def _row_status(self, source_text, target_text, source_missing=False, target_missing=False):
        source_text = self._normalize_review_text(source_text)
        target_text = self._normalize_review_text(target_text)
        if source_missing or target_missing:
            return "red", "dropped/added"
        if not target_text:
            return "red", "empty"
        if source_text and source_text == target_text and self._has_linguistic_letters(source_text):
            return "red", "untranslated"
        if source_text:
            source_len = len(source_text)
            target_len = len(target_text)
            ratio = target_len / max(1, source_len)
            if source_len < 12:
                if target_len > 180:
                    return "yellow", f"density-off ({ratio:.1f}x)"
            elif source_len < 30:
                if target_len > max(220, source_len * 8) or target_len < max(2, int(source_len * 0.10)):
                    return "yellow", f"density-off ({ratio:.1f}x)"
            elif ratio < 0.12 or ratio > 6.0:
                return "yellow", f"density-off ({ratio:.1f}x)"
        return "green", "ok"

    @staticmethod
    def _clear_top_skew_promotions(rows):
        for row in rows or []:
            if row.pop("_top_skew_promoted", False):
                if row.get("_machine_accuracy_promoted"):
                    continue
                row["status"] = "green"
                row["reason"] = "ok"

    @staticmethod
    def _row_expected_comparison_text(row, use_machine_translation=False):
        source_text = str((row or {}).get("source", "") or "").strip()
        if use_machine_translation and not (row or {}).get("tooltip_translation_pending"):
            source_text = str((row or {}).get("tooltip_translation", "") or "").strip() or source_text
        return source_text

    @staticmethod
    def _comparison_tokens(text):
        return [
            token for token in re.findall(r"[a-z0-9']+|[\uac00-\ud7a3]+", str(text or "").lower())
            if re.search(r"[a-z0-9\uac00-\ud7a3]", token)
        ]

    @classmethod
    def _latin_token_overlap(cls, source_text, target_text):
        source_tokens = [
            token for token in cls._comparison_tokens(source_text)
            if re.search(r"[a-z0-9]", token)
        ]
        target_tokens = [
            token for token in cls._comparison_tokens(target_text)
            if re.search(r"[a-z0-9]", token)
        ]
        if not source_tokens or not target_tokens:
            return False, 0.0
        common_tokens = sum((Counter(source_tokens) & Counter(target_tokens)).values())
        return True, common_tokens / max(1, len(source_tokens))

    @classmethod
    def _row_skew_metrics(cls, row, use_machine_translation=False):
        source_text = cls._row_expected_comparison_text(row, use_machine_translation=use_machine_translation)
        target_text = str((row or {}).get("target", "") or "").strip()
        if not source_text or not target_text:
            return None
        if not (row or {}).get("source_tag") or not (row or {}).get("target_tag"):
            return None
        source_len = len(source_text)
        target_len = len(target_text)
        ratio = target_len / max(1, source_len)
        ratio_skew = max(ratio, 1.0 / max(ratio, 0.0001))
        source_tokens = cls._comparison_tokens(source_text)
        target_tokens = cls._comparison_tokens(target_text)
        common_tokens = 0
        if source_tokens and target_tokens:
            common_tokens = sum((Counter(source_tokens) & Counter(target_tokens)).values())
        comparable_token_overlap, source_token_overlap = cls._latin_token_overlap(source_text, target_text)
        unmatched_tokens = max(len(source_tokens), len(target_tokens)) - common_tokens
        similarity = SequenceMatcher(None, source_text.lower(), target_text.lower()).ratio()
        source_weight = max(0.25, min(1.0, source_len / 80.0))
        score = (
            ratio_skew * source_weight
            + min(12.0, abs(target_len - source_len) / 35.0)
            + min(12.0, unmatched_tokens / 2.5)
            + (1.0 - similarity) * 2.0
        )
        return {
            "ratio": ratio,
            "source_len": source_len,
            "source_token_count": len(source_tokens),
            "target_len": target_len,
            "target_token_count": len(target_tokens),
            "comparable_token_overlap": comparable_token_overlap,
            "source_token_overlap": source_token_overlap,
            "score": score,
        }

    @classmethod
    def _row_length_ratio(cls, row, use_machine_translation=False):
        metrics = cls._row_skew_metrics(row, use_machine_translation=use_machine_translation)
        return None if metrics is None else metrics["ratio"]

    def _promote_top_skewed_row_for_count_mismatch(self, rows, source_count, target_count, use_machine_translation=False):
        if source_count == target_count:
            return False
        if not any(row.get("status") == "red" for row in rows or []):
            return False

        best_row = None
        best_ratio = None
        best_score = None
        candidates = []
        for row_index, row in enumerate(rows or []):
            if row.get("status") not in {"green", "yellow"}:
                continue
            metrics = self._row_skew_metrics(row, use_machine_translation=use_machine_translation)
            if metrics is None:
                continue
            candidates.append((row_index, row, metrics))

        if not candidates:
            return False

        target_lengths = sorted(max(1, int(metrics.get("target_len") or 0)) for _index, _row, metrics in candidates)
        target_tokens = sorted(max(1, int(metrics.get("target_token_count") or 0)) for _index, _row, metrics in candidates)
        median_target_len = target_lengths[len(target_lengths) // 2]
        median_target_tokens = target_tokens[len(target_tokens) // 2]
        source_missing_from_output = source_count > target_count
        output_added = target_count > source_count

        for row_index, row, metrics in candidates:
            ratio = metrics["ratio"]
            source_len = max(1, int(metrics.get("source_len") or 0))
            source_token_count = max(1, int(metrics.get("source_token_count") or 0))
            target_len = max(1, int(metrics.get("target_len") or 0))
            target_token_count = max(1, int(metrics.get("target_token_count") or 0))
            target_len_outlier = target_len / max(1, median_target_len)
            target_token_outlier = target_token_count / max(1, median_target_tokens)
            if source_missing_from_output:
                directional_len_delta = max(0, target_len - source_len)
                directional_token_delta = max(0, target_token_count - source_token_count)
                directional_ratio = max(0.0, ratio - 1.0)
                directional_len_ratio = directional_len_delta / max(1, source_len)
                directional_token_ratio = directional_token_delta / max(1, source_token_count)
            elif output_added:
                directional_len_delta = max(0, source_len - target_len)
                directional_token_delta = max(0, source_token_count - target_token_count)
                directional_ratio = max(0.0, (1.0 / max(ratio, 0.0001)) - 1.0)
                directional_len_ratio = directional_len_delta / max(1, target_len)
                directional_token_ratio = directional_token_delta / max(1, target_token_count)
            else:
                directional_len_delta = abs(target_len - source_len)
                directional_token_delta = abs(target_token_count - source_token_count)
                directional_ratio = max(ratio, 1.0 / max(ratio, 0.0001)) - 1.0
                directional_len_ratio = directional_len_delta / max(1, min(source_len, target_len))
                directional_token_ratio = directional_token_delta / max(1, min(source_token_count, target_token_count))
            output_column_weight = (
                target_len_outlier * 6.0
                + target_token_outlier * 4.0
                + min(8.0, target_len / 80.0)
            )
            ratio_multiplier = 1.0 + min(5.0, directional_ratio * 2.0)
            anchor_factor = 1.0
            if metrics.get("comparable_token_overlap") and directional_ratio > 0.0:
                source_token_overlap = float(metrics.get("source_token_overlap") or 0.0)
                anchor_factor = min(1.0, 0.10 + source_token_overlap * 1.20)
            downstream_shift_factor = 1.0
            if source_missing_from_output and metrics.get("comparable_token_overlap") and directional_ratio > 0.0:
                source_token_overlap = float(metrics.get("source_token_overlap") or 0.0)
                next_source_overlap = 0.0
                for next_row in (rows or [])[row_index + 1:row_index + 3]:
                    next_source_text = self._row_expected_comparison_text(
                        next_row,
                        use_machine_translation=use_machine_translation,
                    )
                    comparable, overlap = self._latin_token_overlap(next_source_text, row.get("target"))
                    if comparable:
                        next_source_overlap = max(next_source_overlap, overlap)
                if next_source_overlap >= 0.45 and source_token_overlap < 0.35:
                    downstream_shift_factor = 0.05
            ratio_sensitive_score = (
                min(6.0, directional_len_ratio) * 130.0
                + min(6.0, directional_token_ratio) * 110.0
                + min(6.0, directional_ratio) * 70.0
                + min(240, directional_len_delta) * 0.35
                + min(80, directional_token_delta) * 2.5
                + output_column_weight * ratio_multiplier
            )
            score = (
                ratio_sensitive_score * anchor_factor
                + metrics["score"] * 0.15
            ) * downstream_shift_factor
            if best_score is None or score > best_score:
                best_row = row
                best_ratio = ratio
                best_score = score

        if not best_row or best_score is None or best_row.get("status") != "green":
            return False
        best_row["status"] = "yellow"
        best_row["reason"] = f"top translated-column skew ({best_ratio:.2f}x)"
        best_row["_top_skew_promoted"] = True
        return True

    @staticmethod
    def _clear_machine_accuracy_promotions(rows):
        for row in rows or []:
            if row.pop("_machine_accuracy_promoted", False):
                row["status"] = row.pop("_machine_accuracy_previous_status", "green")
                row["reason"] = row.pop("_machine_accuracy_previous_reason", "ok")
            elif row.get("status") == "purple" and str(row.get("reason") or "").startswith("machine translation inaccurate"):
                row["status"] = "green"
                row["reason"] = "ok"
            row.pop("_machine_accuracy_previous_status", None)
            row.pop("_machine_accuracy_previous_reason", None)

    def _machine_translation_accuracy_score(self, rows, row_index):
        try:
            row = rows[row_index]
        except Exception:
            return None
        if row.get("status") == "red":
            return None
        if row.get("tooltip_translation_pending"):
            return None
        expected = str(row.get("tooltip_translation") or "").strip()
        target = str(row.get("target") or "").strip()
        if not expected or not target:
            return None
        if not row.get("source_tag") or not row.get("target_tag"):
            return None
        expected_normalized = self._normalized_machine_translation_text(expected)
        target_normalized = self._normalized_machine_translation_text(target)
        if expected_normalized == target_normalized:
            return 0.0
        if self._compact_machine_translation_text(expected_normalized) == self._compact_machine_translation_text(target_normalized):
            return 0.0
        expected = expected_normalized
        target = target_normalized

        expected_tokens = self._comparison_tokens(expected)
        target_tokens = self._comparison_tokens(target)
        if self._machine_translation_text_too_short_for_accuracy(expected, target, expected_tokens, target_tokens):
            return 0.0
        if expected_tokens and target_tokens:
            common = sum((Counter(expected_tokens) & Counter(target_tokens)).values())
            token_f1 = (2.0 * common) / max(1, len(expected_tokens) + len(target_tokens))
        else:
            token_f1 = 0.0
        expected_content_tokens = self._machine_translation_content_tokens(expected_tokens)
        target_content_tokens = self._machine_translation_content_tokens(target_tokens)
        content_penalty = 0.0
        if len(expected_content_tokens) >= 4 and len(target_content_tokens) >= 4:
            content_common = sum((Counter(expected_content_tokens) & Counter(target_content_tokens)).values())
            content_f1 = (2.0 * content_common) / max(1, len(expected_content_tokens) + len(target_content_tokens))
            if content_f1 < 0.45:
                content_penalty = (0.45 - content_f1) * 165.0
        similarity = SequenceMatcher(None, expected.lower(), target.lower()).ratio()
        expected_len = max(1, len(expected))
        target_len = max(1, len(target))
        length_ratio = max(expected_len, target_len) / max(1, min(expected_len, target_len))
        expected_token_count = max(1, len(expected_tokens))
        target_token_count = max(1, len(target_tokens))
        token_ratio = max(expected_token_count, target_token_count) / max(1, min(expected_token_count, target_token_count))
        return (
            (1.0 - token_f1) * 120.0
            + (1.0 - similarity) * 70.0
            + min(6.0, length_ratio - 1.0) * 55.0
            + min(6.0, token_ratio - 1.0) * 45.0
            + content_penalty
        )

    @classmethod
    def _machine_translation_text_too_short_for_accuracy(cls, expected, target, expected_tokens=None, target_tokens=None):
        expected_tokens = expected_tokens if expected_tokens is not None else cls._comparison_tokens(expected)
        target_tokens = target_tokens if target_tokens is not None else cls._comparison_tokens(target)
        if not expected_tokens or not target_tokens:
            return False
        if (
            len(expected_tokens) != cls.MACHINE_TRANSLATION_SHORT_TEXT_MAX_TOKENS
            or len(target_tokens) != cls.MACHINE_TRANSLATION_SHORT_TEXT_MAX_TOKENS
        ):
            return False
        expected_len = len(cls._compact_machine_translation_text(expected))
        target_len = len(cls._compact_machine_translation_text(target))
        return max(expected_len, target_len) <= cls.MACHINE_TRANSLATION_SHORT_TEXT_MAX_CHARS

    @classmethod
    def _machine_translation_content_tokens(cls, tokens):
        return [
            token for token in (tokens or [])
            if token not in cls.MACHINE_TRANSLATION_CONTENT_STOPWORDS
        ]

    def _promote_inaccurate_machine_translation_rows(self, piece, threshold=None):
        rows = piece.get("rows") or []
        if not rows:
            return None
        threshold = float(threshold if threshold is not None else self._machine_translation_inaccuracy_threshold())
        self._clear_machine_accuracy_promotions(rows)
        self._clear_top_skew_promotions(rows)

        scored_rows = []
        promoted_indices = []
        for row_index, _row in enumerate(rows):
            score = self._machine_translation_accuracy_score(rows, row_index)
            if score is None:
                continue
            scored_rows.append((row_index, score))
            if score >= threshold:
                promoted_indices.append((row_index, score))

        if not scored_rows:
            piece.pop("_machine_accuracy_review_active", None)
            return None

        piece["_machine_accuracy_review_active"] = True
        for row_index, score in promoted_indices:
            row = rows[row_index]
            row["_machine_accuracy_previous_status"] = row.get("status", "green")
            row["_machine_accuracy_previous_reason"] = row.get("reason", "ok")
            row["_machine_accuracy_promoted"] = True
            row["status"] = "purple"
            row["reason"] = f"machine translation inaccurate ({score:.0f} >= {threshold:.0f})"
        return [row_index for row_index, _score in promoted_indices]

    def _flag_current_piece_inaccurate_translations(self):
        row = self._displayed_piece_row()
        if row < 0 or row >= len(self.pieces):
            return
        self._start_flag_accuracy_button_animation()
        try:
            piece = self.pieces[row]
            rows = piece.get("rows") or []
            before = [
                (str(row_data.get("status") or ""), str(row_data.get("reason") or ""))
                for row_data in rows
            ]
            promoted_indices = self._promote_inaccurate_machine_translation_rows(piece)
            if promoted_indices is None:
                try:
                    self.save_status_label.setText("Generate Machine Translation Preview first")
                except Exception:
                    pass
                return

            self._refresh_piece_summary(piece)
            changed_rows = [
                row_index for row_index, row_data in enumerate(rows)
                if row_index >= len(before)
                or before[row_index] != (str(row_data.get("status") or ""), str(row_data.get("reason") or ""))
            ]
            self._invalidate_piece_render_model(piece, restart_preload=False)
            self._refresh_piece_list_item(row)
            self._refresh_piece_header(row)
            for row_index in changed_rows:
                self._refresh_visible_review_row_status(row, row_index)
            try:
                if promoted_indices:
                    self.save_status_label.setText(f"Flagged {len(promoted_indices)} inaccurate machine translation row(s)")
                else:
                    self.save_status_label.setText("No inaccurate machine translation rows found")
            except Exception:
                pass
        finally:
            self._queue_stop_flag_accuracy_button_animation()

    @staticmethod
    def _heading_tag_level_changed(source_tag, target_tag):
        source_tag = str(source_tag or "").strip().lower()
        target_tag = str(target_tag or "").strip().lower()
        if source_tag == target_tag:
            return False
        return bool(re.fullmatch(r"h[1-6]", source_tag) and re.fullmatch(r"h[1-6]", target_tag))

    @staticmethod
    def _heading_paragraph_tag_changed(source_tag, target_tag):
        source_tag = str(source_tag or "").strip().lower()
        target_tag = str(target_tag or "").strip().lower()
        if source_tag == target_tag:
            return False
        source_heading = bool(re.fullmatch(r"h[1-6]", source_tag))
        target_heading = bool(re.fullmatch(r"h[1-6]", target_tag))
        text_block_tags = {"p", "li"}
        return (
            (source_heading and target_tag in text_block_tags)
            or (source_tag in text_block_tags and target_heading)
        )

    @staticmethod
    def _paragraph_list_item_tag_changed(source_tag, target_tag):
        source_tag = str(source_tag or "").strip().lower()
        target_tag = str(target_tag or "").strip().lower()
        return {source_tag, target_tag} == {"p", "li"}

    def _tag_mismatch_status(self, source_tag, target_tag):
        if self._paragraph_list_item_tag_changed(source_tag, target_tag):
            return "green", "paragraph/list-item text unit"
        if self._heading_tag_level_changed(source_tag, target_tag):
            return "yellow", "heading level changed"
        if self._heading_paragraph_tag_changed(source_tag, target_tag):
            return "yellow", "heading/paragraph tag changed"
        return "red", "tag mismatch"

    @staticmethod
    def _review_unit_is_heading(unit):
        tag = str((unit or {}).get("tag", "") or "").strip().lower()
        return bool(re.fullmatch(r"h[1-6]", tag))

    @staticmethod
    def _review_unit_is_paragraph(unit):
        return str((unit or {}).get("tag", "") or "").strip().lower() in {"p", "li"}

    def _review_units_are_compatible(self, source_unit, target_unit):
        source_tag = str((source_unit or {}).get("tag", "") or "").strip().lower()
        target_tag = str((target_unit or {}).get("tag", "") or "").strip().lower()
        if source_tag == target_tag:
            return True
        return (
            self._paragraph_list_item_tag_changed(source_tag, target_tag)
            or self._heading_tag_level_changed(source_tag, target_tag)
            or self._heading_paragraph_tag_changed(source_tag, target_tag)
        )

    def _align_review_units(self, source_units, target_units):
        source_units = list(source_units or [])
        target_units = list(target_units or [])
        rows = []
        i = 0
        j = 0
        # Track the most recently consumed unit on each side (and the most
        # recent source heading) so we can recognise a side-only "echo" — a
        # unit that simply repeats an earlier line and has no counterpart on
        # the other side.
        last_src = None
        last_tgt = None
        last_src_heading = None

        def _same_text(unit_a, unit_b):
            if not unit_a or not unit_b:
                return False
            return (
                self._normalize_review_text(unit_a.get("text", ""))
                == self._normalize_review_text(unit_b.get("text", ""))
            )

        while i < len(source_units) or j < len(target_units):
            src = source_units[i] if i < len(source_units) else None
            tgt = target_units[j] if j < len(target_units) else None
            if tgt is not None and tgt.get("user_added"):
                rows.append((None, tgt))
                last_tgt = tgt
                j += 1
                continue
            if src is None:
                rows.append((None, tgt))
                last_tgt = tgt
                j += 1
                continue
            if tgt is None:
                rows.append((src, None))
                last_src = src
                if self._review_unit_is_heading(src):
                    last_src_heading = src
                i += 1
                continue
            next_src = source_units[i + 1] if i + 1 < len(source_units) else None
            next_tgt = target_units[j + 1] if j + 1 < len(target_units) else None
            source_remaining = len(source_units) - i
            target_remaining = len(target_units) - j

            # A user can turn a formerly empty target container into a real
            # text unit in rendered Notepad mode. If the next target is the
            # current source text, the current target is an insertion, not the
            # translation for this source row. Emit it as target-only so every
            # following source/target pair keeps its correct position.
            if target_remaining > source_remaining:
                target_surplus = target_remaining - source_remaining
                for lookahead in range(1, min(target_surplus, len(target_units) - j - 1) + 1):
                    future_target = target_units[j + lookahead]
                    if _same_text(src, future_target) and self._review_units_are_compatible(src, future_target):
                        rows.append((None, tgt))
                        last_tgt = tgt
                        j += 1
                        break
                else:
                    lookahead = 0
                if lookahead:
                    continue

            if source_remaining > target_remaining:
                source_surplus = source_remaining - target_remaining
                for lookahead in range(1, min(source_surplus, len(source_units) - i - 1) + 1):
                    future_source = source_units[i + lookahead]
                    if _same_text(future_source, tgt) and self._review_units_are_compatible(future_source, tgt):
                        rows.append((src, None))
                        last_src = src
                        if self._review_unit_is_heading(src):
                            last_src_heading = src
                        i += 1
                        break
                else:
                    lookahead = 0
                if lookahead:
                    continue

            if self._review_unit_is_heading(src) and self._review_unit_is_paragraph(tgt):
                if source_remaining > target_remaining and next_src is not None and self._review_units_are_compatible(next_src, tgt):
                    rows.append((src, None))
                    last_src = src
                    last_src_heading = src
                    i += 1
                    continue

            if self._review_unit_is_paragraph(src) and self._review_unit_is_heading(tgt):
                if target_remaining > source_remaining and next_tgt is not None and self._review_units_are_compatible(src, next_tgt):
                    rows.append((None, tgt))
                    last_tgt = tgt
                    j += 1
                    continue

            # The source carries a surplus unit the target lacks — most often
            # the chapter title echoed as the first <p> (the heading and the
            # echo are both present in the source, but the cleaned output keeps
            # the title only once). Both sides are usually <p> here, so the
            # tag-based checks above can't see it; pairing 1:1 would shift every
            # following row's source/output + machine-translation pairing off by
            # one. Detect the surplus unit as an echo of an earlier source line
            # (its predecessor or the most recent heading) and emit a gap so the
            # rest of the chapter stays aligned.
            if source_remaining > target_remaining:
                src_is_echo = _same_text(src, last_src) or _same_text(src, last_src_heading)
                tgt_is_echo = _same_text(tgt, last_tgt)
                if src_is_echo and not tgt_is_echo:
                    rows.append((src, None))
                    last_src = src
                    i += 1
                    continue

            # Symmetric case: the target repeats a line the source does not.
            if target_remaining > source_remaining:
                tgt_is_echo = _same_text(tgt, last_tgt)
                src_is_echo = _same_text(src, last_src)
                if tgt_is_echo and not src_is_echo:
                    rows.append((None, tgt))
                    last_tgt = tgt
                    j += 1
                    continue

            rows.append((src, tgt))
            last_src = src
            last_tgt = tgt
            if self._review_unit_is_heading(src):
                last_src_heading = src
            i += 1
            j += 1
        return rows

    def _infer_user_added_empty_target_indexes(
        self,
        source_all_units,
        target_all_units,
    ):
        """Recover legacy empty Notepad paragraphs created before index metadata."""
        source_all_units = list(source_all_units or [])
        target_all_units = list(target_all_units or [])
        surplus = len(target_all_units) - len(source_all_units)
        if surplus <= 0:
            return set()
        inferred = set()
        source_index = 0
        target_index = 0
        while target_index < len(target_all_units) and surplus > 0:
            target_unit = target_all_units[target_index]
            source_unit = (
                source_all_units[source_index]
                if source_index < len(source_all_units)
                else None
            )
            target_empty = not self._normalize_review_text(
                target_unit.get("text", "")
            )
            source_empty = bool(
                source_unit is not None
                and not self._normalize_review_text(source_unit.get("text", ""))
            )
            if target_empty and (source_unit is None or not source_empty):
                inferred.add(target_unit.get("index"))
                target_index += 1
                surplus -= 1
                continue
            source_index += 1
            target_index += 1
        return {value for value in inferred if isinstance(value, int) and value >= 0}

    def _align_review_units_by_dom_position(
        self,
        source_all_units,
        target_all_units,
        source_review_units,
        target_review_units,
        *,
        include_empty_target=False,
    ):
        """Align equal HTML text-unit structures without collapsing empty slots.

        Empty slots are important in the rendered editor: when a user types in
        one, it becomes a target-only Added row. Positional alignment prevents
        that new row from stealing the next source paragraph.
        """
        source_all_units = list(source_all_units or [])
        target_all_units = list(target_all_units or [])
        if any(unit.get("user_added") for unit in target_review_units or []):
            # A recorded user insertion is a real target-only slot even when
            # the total source/target counts happen to match after some other
            # deletion. Positional zip alignment must never consume a source
            # row with that inserted paragraph.
            return None
        if not source_all_units or len(source_all_units) != len(target_all_units):
            return None
        if not all(
            self._review_units_are_compatible(source_unit, target_unit)
            for source_unit, target_unit in zip(source_all_units, target_all_units)
        ):
            return None

        source_by_index = {
            unit.get("index"): unit
            for unit in source_review_units or []
            if isinstance(unit, dict)
        }
        target_by_index = {
            unit.get("index"): unit
            for unit in target_review_units or []
            if isinstance(unit, dict)
        }
        rows = []
        for source_all, target_all in zip(source_all_units, target_all_units):
            source_index = source_all.get("index")
            target_index = target_all.get("index")
            source_unit = source_by_index.get(source_index)
            target_unit = target_by_index.get(target_index)
            if target_unit is not None and not str(target_unit.get("text") or "").strip():
                if not include_empty_target or source_unit is None:
                    target_unit = None
            if source_unit is not None or target_unit is not None:
                rows.append((source_unit, target_unit))
        return rows

    @staticmethod
    def _review_piece_non_empty_count(rows, side):
        text_key = "source" if side == "source" else "target"
        tag_key = "source_tag" if side == "source" else "target_tag"
        try:
            return sum(
                1 for row in (rows or [])
                if row.get(tag_key) and str(row.get(text_key, "") or "").strip()
            )
        except Exception:
            return 0

    def _refresh_piece_summary(self, piece):
        rows = piece.get("rows") or []
        source_count = self._review_piece_non_empty_count(rows, "source")
        target_count = self._review_piece_non_empty_count(rows, "target")
        manual_green_override = bool(piece.get("manual_green_override"))
        self._clear_top_skew_promotions(rows)
        manual_accuracy_active = (
            bool(piece.get("_machine_accuracy_review_active"))
            or any(row.get("_machine_accuracy_promoted") for row in rows)
        )
        if not manual_accuracy_active and not manual_green_override:
            self._promote_top_skewed_row_for_count_mismatch(rows, source_count, target_count)
        if manual_green_override:
            reason = str(piece.get("manual_green_reason") or "manually marked completed")
            for row in rows:
                if row.get("status") in self.MANUAL_GREEN_STATUSES:
                    row["status"] = "green"
                    row["reason"] = reason
                    row.pop("_top_skew_promoted", None)
        red_count = sum(1 for row in rows if row.get("status") == "red")
        yellow_count = sum(1 for row in rows if row.get("status") == "yellow")
        purple_count = sum(1 for row in rows if row.get("status") == "purple")
        piece["source_count"] = source_count
        piece["target_count"] = target_count
        piece["red_count"] = red_count
        piece["yellow_count"] = yellow_count
        piece["purple_count"] = purple_count
        piece["count_ratio"] = (target_count / source_count) if source_count else (1.0 if not target_count else 0.0)
        piece["mismatch"] = False if manual_green_override else (source_count != target_count or red_count > 0)
        return piece

    @staticmethod
    def _canonical_basename(name):
        return os.path.basename(str(name or "").replace("\\", "/")).lower()

    @staticmethod
    def _canonical_review_path(name):
        text = str(name or "").replace("\\", "/").strip().lower()
        text = text.lstrip("/")
        while text.startswith("./"):
            text = text[2:]
        while text.startswith("../"):
            text = text[3:]
        return text

    @staticmethod
    def _sidecar_output_name(path_or_name):
        sidecar_name = os.path.basename(str(path_or_name or "").replace("\\", "/"))
        suffix = ".sdlxliff"
        if sidecar_name.lower().endswith(suffix):
            return sidecar_name[:-len(suffix)]
        return sidecar_name

    @staticmethod
    def _chapter_number_from_name(name):
        matches = re.findall(r"(\d+)", str(name or ""))
        if not matches:
            return 0
        try:
            return int(matches[-1])
        except Exception:
            return 0

    @staticmethod
    def _format_chapter_number(value):
        try:
            if isinstance(value, float):
                return f"{int(value):03d}" if value.is_integer() else f"{value:06.1f}"
            return f"{int(value):03d}"
        except Exception:
            return "000"

    def _read_progress_metadata(self):
        progress_path = os.path.join(self.output_dir or "", "translation_progress.json")
        output_map = {}
        if not os.path.isfile(progress_path):
            return output_map
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
        except Exception:
            return output_map

        chapters = progress_data.get("chapters") if isinstance(progress_data, dict) else None
        if not isinstance(chapters, dict):
            chapters = progress_data if isinstance(progress_data, dict) else {}

        for key, entry in chapters.items():
            if not isinstance(entry, dict):
                continue
            output_file = entry.get("output_file")
            if not output_file:
                continue
            normalized = self._canonical_basename(output_file)
            if not normalized:
                continue
            copied = dict(entry)
            copied["_progress_key"] = key
            output_map[normalized] = copied
            logical = _normalize_progress_match_name(output_file).casefold()
            if logical:
                output_map.setdefault(f"logical:{logical}", copied)
        return output_map

    def _find_opf_path(self, allow_deep_search=True):
        output_dir = self.output_dir or ""
        return find_workspace_opf_path(
            output_dir,
            recursive=allow_deep_search,
        )

    def _read_spine_positions(self, allow_deep_search=True):
        opf_path = self._find_opf_path(allow_deep_search=allow_deep_search)
        spine_positions = {}
        if opf_path:
            try:
                root = ET.parse(opf_path).getroot()
            except Exception:
                root = None
            if root is not None:
                id_to_href = {}
                for element in root.iter():
                    if self._local_name(element.tag) != "item":
                        continue
                    item_id = element.attrib.get("id")
                    href = element.attrib.get("href")
                    if item_id and href:
                        id_to_href[item_id] = unquote(href)

                position = 0
                for element in root.iter():
                    if self._local_name(element.tag) != "itemref":
                        continue
                    idref = element.attrib.get("idref")
                    href = id_to_href.get(idref)
                    if not href:
                        continue
                    href_key = self._canonical_review_path(href)
                    if href_key:
                        spine_positions[href_key] = position
                        no_ext, _ext = os.path.splitext(href_key)
                        if no_ext:
                            spine_positions[no_ext] = position
                    href_base = self._canonical_basename(href)
                    if href_base:
                        spine_positions[href_base] = position
                        no_ext, _ext = os.path.splitext(href_base)
                        if no_ext:
                            spine_positions[no_ext] = position
                    position += 1

        # Manual generation reads source-EPUB OPF data in its background
        # worker. Reuse that cache here when content.opf is not copied into the
        # output folder, without doing ZIP I/O on the GUI thread.
        owner = getattr(self, "_sdlxliff_autogen_owner", None)
        cached_positions = getattr(owner, "_sdlxliff_cached_source_spine_positions", None)
        if isinstance(cached_positions, dict):
            for key, position in cached_positions.items():
                spine_positions.setdefault(str(key), position)
        return spine_positions

    def _original_name_from_output(self, output_name):
        name = os.path.basename(str(output_name or ""))
        if name.lower().startswith("response_"):
            name = name[len("response_"):]
        root, ext = os.path.splitext(name)
        if ext.lower() in (".html", ".htm"):
            return f"{root}.xhtml"
        return name

    def _sidecar_metadata(self, path, fallback_index, progress_map, spine_positions):
        output_name = self._sidecar_output_name(path)
        progress_entry = progress_map.get(self._canonical_basename(output_name), {})
        if not progress_entry:
            progress_entry = progress_map.get(
                f"logical:{_normalize_progress_match_name(output_name).casefold()}",
                {},
            )
        original_name = (
            progress_entry.get("original_basename")
            or progress_entry.get("original_filename")
            or self._original_name_from_output(output_name)
        )

        # content.opf is authoritative. Progress positions can be stale or can
        # reflect translation/insertion order rather than EPUB reading order.
        opf_position = None
        for candidate in (original_name, output_name, self._original_name_from_output(output_name)):
            if opf_position is not None:
                break
            normalized_keys = [
                self._canonical_review_path(candidate),
                self._canonical_basename(candidate),
            ]
            for normalized in normalized_keys:
                if not normalized:
                    continue
                if normalized in spine_positions:
                    opf_position = spine_positions[normalized]
                    break
                no_ext, _ext = os.path.splitext(normalized)
                if no_ext in spine_positions:
                    opf_position = spine_positions[no_ext]
                    break
            if opf_position is not None:
                break

        if opf_position is None:
            opf_position = progress_entry.get("opf_position")
            if opf_position is None:
                opf_position = progress_entry.get("position")
            try:
                opf_position = int(opf_position) if opf_position is not None else None
            except Exception:
                opf_position = None

        chapter_num = progress_entry.get("actual_num", progress_entry.get("chapter_num"))
        source_stem = os.path.splitext(os.path.basename(str(original_name or "")))[0]
        is_special = bool(progress_entry.get("is_special")) or _is_special_basename(original_name)
        if is_special or not re.search(r"\d", source_stem):
            chapter_num = 0
        elif chapter_num is None:
            chapter_num = self._chapter_number_from_name(output_name)

        sort_position = opf_position if opf_position is not None else 100000 + fallback_index
        metadata = {
            "output_name": output_name,
            "progress_entry": progress_entry,
            "progress_key": progress_entry.get("_progress_key"),
            "original_name": original_name,
            "opf_position": opf_position,
            "chapter_num": chapter_num,
            "sort_key": (sort_position, self._chapter_number_from_name(output_name), output_name.lower()),
        }
        return metadata

    def _review_label_from_metadata(self, metadata):
        display_position = metadata.get("display_position")
        if display_position is None:
            opf_position = metadata.get("opf_position")
            display_position = (int(opf_position) + 1) if opf_position is not None else 1
        chapter_label = self._format_chapter_number(
            metadata.get("display_chapter_num", metadata.get("chapter_num"))
        )
        return f"[{int(display_position):03d}] Ch.{chapter_label} |"

    def _sidebar_label_for_piece(self, piece, row):
        prefix = piece.get("review_label") or f"[{row + 1:03d}] Ch.{self._format_chapter_number(piece.get('chapter_num'))} |"
        return f"{prefix} {piece.get('source_count', 0)} -> {piece.get('target_count', 0)}"

    def _build_piece(self, path, index, metadata=None):
        trace_started = time.perf_counter()
        metadata = dict(metadata or {})
        metadata.setdefault("output_name", self._sidecar_output_name(path))
        metadata.setdefault("display_position", index + 1)
        metadata.setdefault("label", self._review_label_from_metadata(metadata))
        try:
            (
                source_html,
                target_html,
                user_added_target_indexes,
                user_added_break_positions,
            ) = (
                self._read_sdlxliff_html_pair(
                    path,
                    include_user_added_indexes=True,
                )
            )
            manual_untranslated = _is_manual_untranslated_sdlxliff(path)
            manual_editing = (
                manual_untranslated
                or _is_manual_editing_sdlxliff(path)
            )
            source_all_units = self._extract_text_units(source_html, include_empty=True)
            target_all_units = self._extract_text_units(target_html, include_empty=True)
            if manual_editing and not user_added_target_indexes:
                user_added_target_indexes.update(
                    self._infer_user_added_empty_target_indexes(
                        source_all_units,
                        target_all_units,
                    )
                )
            for target_unit in target_all_units:
                if target_unit.get("index") in user_added_target_indexes:
                    target_unit["user_added"] = True
            source_units = self._non_empty_text_units(source_all_units)
            target_units = (
                target_all_units
                if manual_editing
                else [
                    unit for unit in target_all_units
                    if self._normalize_review_text(unit.get("text", ""))
                    or unit.get("user_added")
                ]
            )
            if self._review_remove_duplicate_h1_p_enabled():
                # Hide the same duplicate H1-H6 + <p> title echoes the header
                # translation pipeline removes, so the surplus source unit no
                # longer shows up as a flagged "Empty" row.
                source_all_units = self._dedupe_heading_paragraph_units(source_all_units)
                source_units = self._non_empty_text_units(source_all_units)
                if not manual_editing:
                    target_all_units = self._dedupe_heading_paragraph_units(target_all_units)
                    target_units = self._non_empty_text_units(target_all_units)
            source_review_units = self._annotate_review_tag_labels(self._non_empty_text_units(source_units))
            # An explicitly Notepad-created target block is a translator note
            # regardless of whether the SDLXLIFF itself is in manual-editing
            # mode. Feed it into the one existing TN(N) annotation path.
            translator_note_target_indexes = set(user_added_target_indexes)
            if (
                len(source_all_units) == len(target_all_units)
                and all(
                    self._review_units_are_compatible(source_unit, target_unit)
                    for source_unit, target_unit in zip(
                        source_all_units, target_all_units
                    )
                )
            ):
                translator_note_target_indexes.update({
                    target_unit.get("index")
                    for source_unit, target_unit in zip(
                        source_all_units, target_all_units
                    )
                    if not self._normalize_review_text(source_unit.get("text", ""))
                    and self._normalize_review_text(target_unit.get("text", ""))
                })
            if manual_editing:
                source_review_indexes = {
                    unit.get("index") for unit in source_review_units
                }
                target_review_units = [
                    unit for unit in target_units
                    if str(unit.get("text") or "").strip()
                    or unit.get("index") in source_review_indexes
                    or unit.get("user_added")
                ]
            else:
                target_review_units = [
                    unit for unit in target_units
                    if self._normalize_review_text(unit.get("text", ""))
                    or unit.get("user_added")
                ]

            # Positional alignment must retain an empty target element even
            # for an ordinary translated sidecar. Notepad edits leave the
            # element itself in the DOM when its text is erased; dropping that
            # empty unit here made the reviewer lose its target tag/index and
            # made the row appear deleted instead of red. Keep the older
            # non-empty list for fallback alignment when the two DOMs differ.
            positional_target_review_units = target_review_units
            if not manual_editing:
                source_review_indexes = {
                    unit.get("index") for unit in source_review_units
                }
                positional_target_review_units = [
                    unit for unit in target_all_units
                    if str(unit.get("text") or "").strip()
                    or unit.get("index") in source_review_indexes
                    or unit.get("user_added")
                ]

            target_annotation_units = positional_target_review_units

            def _annotate_target_units():
                for unit in target_annotation_units:
                    unit.pop("tag_label", None)
                    unit.pop("tag_ordinal", None)
                    unit.pop("translator_note", None)
                    unit.pop("translator_note_ordinal", None)
                self._annotate_review_tag_labels([
                    unit for unit in target_annotation_units
                    if unit.get("index") not in translator_note_target_indexes
                ])
                translator_note_ordinal = 0
                for unit in target_annotation_units:
                    if unit.get("index") not in translator_note_target_indexes:
                        continue
                    translator_note_ordinal += 1
                    unit["tag_label"] = f"TN({translator_note_ordinal})"
                    unit["tag_ordinal"] = None
                    unit["translator_note"] = True
                    unit["translator_note_ordinal"] = translator_note_ordinal

            # Text added where the source has no text is a translator note,
            # not a paragraph translation unit. Preserve its DOM index for
            # write-back without letting its <p> consume a p ordinal.
            _annotate_target_units()
            rows = []
            red_count = 0
            yellow_count = 0
            purple_count = 0
            aligned_units = self._align_review_units_by_dom_position(
                source_all_units,
                target_all_units,
                source_review_units,
                positional_target_review_units,
                include_empty_target=True,
            )
            if aligned_units is None:
                target_annotation_units = target_review_units
                _annotate_target_units()
                if manual_editing:
                    # Preserve the original manual-sidecar behavior when the
                    # two DOMs genuinely differ, while retaining non-empty
                    # target-only units so user additions remain visible.
                    if user_added_target_indexes:
                        aligned_target_units = target_review_units
                    else:
                        target_units_by_index = {
                            unit.get("index"): unit
                            for unit in target_review_units
                            if isinstance(unit, dict)
                        }
                        source_indexes = {
                            unit.get("index") for unit in source_review_units
                        }
                        aligned_target_units = [
                            target_units_by_index[source_unit.get("index")]
                            for source_unit in source_review_units
                            if source_unit.get("index") in target_units_by_index
                        ]
                        aligned_target_units.extend(
                            unit for unit in target_review_units
                            if unit.get("index") not in source_indexes
                            and str(unit.get("text") or "").strip()
                        )
                else:
                    aligned_target_units = target_review_units
                aligned_units = self._align_review_units(
                    source_review_units,
                    aligned_target_units,
                )
            if manual_editing:
                # Pressing Enter in the rendered editor can create a genuinely
                # new target <p>, rather than filling an existing empty one.
                # Alignment correctly emits it as target-only; classify that
                # node as a translator note as well, then recalculate the real
                # target tag ordinals without allowing it to shift them.
                translator_note_target_indexes.update(
                    tgt.get("index")
                    for src, tgt in aligned_units
                    if src is None and tgt is not None
                )
                _annotate_target_units()
            for row_idx, (src, tgt) in enumerate(aligned_units):
                translator_note = bool(
                    tgt is not None and tgt.get("translator_note")
                )
                translator_note_label = (
                    str(tgt.get("tag_label") or "") if translator_note else ""
                )
                source_missing = src is None and not translator_note
                target_missing = tgt is None
                status, reason = self._row_status(
                    src.get("text") if src else "",
                    tgt.get("text") if tgt else "",
                    source_missing=source_missing,
                    target_missing=target_missing,
                )
                if translator_note:
                    reason = "translator note"
                if src is not None and tgt is not None and src.get("tag") != tgt.get("tag"):
                    status, reason = self._tag_mismatch_status(src.get("tag"), tgt.get("tag"))
                if status == "red":
                    red_count += 1
                elif status == "yellow":
                    yellow_count += 1
                rows.append({
                    "row_index": row_idx,
                    "source_tag": src.get("tag", "") if src else "",
                    "source_tag_label": src.get("tag_label", "") if src else translator_note_label,
                    "source_tag_ordinal": src.get("tag_ordinal") if src else None,
                    "source": src.get("text", "") if src else "",
                    "source_index": src.get("index") if src else None,
                    "target_tag": "" if translator_note else (tgt.get("tag", "") if tgt else ""),
                    "target_dom_tag": tgt.get("tag", "") if tgt else "",
                    "target_tag_label": translator_note_label if translator_note else (tgt.get("tag_label", "") if tgt else ""),
                    "target_tag_ordinal": None if translator_note else (tgt.get("tag_ordinal") if tgt else None),
                    "target": tgt.get("text", "") if tgt else "",
                    "target_original": tgt.get("text", "") if tgt else "",
                    "target_index": tgt.get("index") if tgt else None,
                    "source_missing": source_missing,
                    "target_missing": target_missing,
                    "translator_note": translator_note,
                    "translator_note_ordinal": (
                        tgt.get("translator_note_ordinal")
                        if translator_note else None
                    ),
                    "status": status,
                    "reason": reason,
                })
            source_count = self._review_piece_non_empty_count(rows, "source")
            target_count = self._review_piece_non_empty_count(rows, "target")
            count_ratio = (target_count / source_count) if source_count else (1.0 if not target_count else 0.0)
            piece = {
                "path": path,
                "index": index,
                "name": os.path.basename(path),
                "output_name": metadata.get("output_name"),
                "original_name": metadata.get("original_name"),
                "progress_key": metadata.get("progress_key"),
                "review_label": metadata.get("label"),
                "opf_position": metadata.get("opf_position"),
                "chapter_num": metadata.get("chapter_num"),
                "raw_chapter_num": metadata.get(
                    "raw_chapter_num", metadata.get("chapter_num")
                ),
                "display_chapter_num": metadata.get("display_chapter_num"),
                "sort_key": metadata.get("sort_key"),
                "source_html": source_html,
                "target_html": target_html,
                "user_added_target_indexes": sorted(user_added_target_indexes),
                "user_added_break_positions": dict(user_added_break_positions),
                "source_count": source_count,
                "target_count": target_count,
                "count_ratio": count_ratio,
                "red_count": red_count,
                "yellow_count": yellow_count,
                "purple_count": purple_count,
                "mismatch": source_count != target_count or red_count > 0,
                "manual_editing": manual_editing,
                "manual_untranslated": manual_untranslated,
                "manual_editing_pending": bool(
                    (metadata.get("progress_entry") or {}).get("manual_editing_pending")
                ),
                "rows": rows,
            }
            self._load_machine_translation_file_for_piece(piece)
            self._refresh_piece_summary(piece)
            self._apply_persisted_manual_green_override(piece)
            self._trace_review_perf(
                "build_piece",
                trace_started,
                index=index,
                source=source_count,
                target=target_count,
                output=metadata.get("output_name") or os.path.basename(path),
            )
            return piece
        except Exception as exc:
            self._trace_review_perf(
                "build_piece_failed",
                trace_started,
                force=True,
                index=index,
                output=metadata.get("output_name") or os.path.basename(path),
                error=str(exc)[:160],
            )
            return {
                "path": path,
                "index": index,
                "name": os.path.basename(path),
                "output_name": metadata.get("output_name"),
                "original_name": metadata.get("original_name"),
                "progress_key": metadata.get("progress_key"),
                "review_label": metadata.get("label"),
                "opf_position": metadata.get("opf_position"),
                "chapter_num": metadata.get("chapter_num"),
                "raw_chapter_num": metadata.get(
                    "raw_chapter_num", metadata.get("chapter_num")
                ),
                "display_chapter_num": metadata.get("display_chapter_num"),
                "sort_key": metadata.get("sort_key"),
                "source_count": 0,
                "target_count": 0,
                "count_ratio": 0.0,
                "red_count": 1,
                "yellow_count": 0,
                "purple_count": 0,
                "mismatch": True,
                "manual_editing": _is_manual_editing_sdlxliff(path),
                "manual_untranslated": _is_manual_untranslated_sdlxliff(path),
                "manual_editing_pending": bool(
                    (metadata.get("progress_entry") or {}).get("manual_editing_pending")
                ),
                "error": str(exc),
                "rows": [],
            }

    @staticmethod
    def _review_piece_is_empty_sidecar(piece):
        if not isinstance(piece, dict) or piece.get("error"):
            return False
        return int(piece.get("source_count") or 0) == 0 and int(piece.get("target_count") or 0) == 0

    def _filter_review_pieces(self, pieces):
        filtered = [
            piece for piece in (pieces or [])
            if piece is not None and not self._review_piece_is_empty_sidecar(piece)
        ]
        for index, piece in enumerate(filtered):
            piece["index"] = index
        return filtered

    def _deduplicate_review_sidecar_paths(self, paths, progress_map=None):
        """Choose one sidecar for each logical chapter without deleting files.

        A filename-mode change can leave both ``response_chapter.html`` and
        ``chapter.xhtml`` sidecars behind. They describe one Progress Manager
        entry and must occupy one spine position in the reviewer.
        """
        progress_map = progress_map if isinstance(progress_map, dict) else {}
        current_norm = ""
        try:
            if self.current_path:
                current_norm = os.path.normcase(os.path.abspath(self.current_path))
        except Exception:
            current_norm = ""

        grouped = {}
        passthrough = []
        for order, path in enumerate(paths or []):
            logical_key = _sdlxliff_logical_output_key(path)
            if logical_key:
                grouped.setdefault(logical_key, []).append((order, path))
            else:
                passthrough.append((order, path))

        selected = list(passthrough)
        for logical_key, candidates in grouped.items():
            if len(candidates) == 1:
                selected.append(candidates[0])
                continue
            progress_entry = progress_map.get(f"logical:{logical_key}", {})
            expected_name = self._canonical_basename(
                progress_entry.get("output_file") if isinstance(progress_entry, dict) else ""
            )

            def _candidate_score(candidate):
                order, path = candidate
                output_name = self._canonical_basename(self._sidecar_output_name(path))
                try:
                    path_norm = os.path.normcase(os.path.abspath(path))
                except Exception:
                    path_norm = ""
                try:
                    edited = not _is_manual_untranslated_sdlxliff(path)
                except Exception:
                    edited = False
                try:
                    modified = os.path.getmtime(path)
                except OSError:
                    modified = 0.0
                return (
                    edited,
                    path_norm == current_norm,
                    bool(expected_name and output_name == expected_name),
                    modified,
                    -order,
                )

            selected.append(max(candidates, key=_candidate_score))

        selected.sort(key=lambda item: item[0])
        return [path for _order, path in selected]

    def _load_pieces(self, stream_sidebar=False):
        trace_started = time.perf_counter()
        sidecar_dir = os.path.join(self.output_dir or "", "SDLXLIFF")
        paths = []
        try:
            if os.path.isdir(sidecar_dir):
                for fname in os.listdir(sidecar_dir):
                    if fname.lower().endswith(".sdlxliff"):
                        paths.append(os.path.join(sidecar_dir, fname))
        except Exception:
            paths = []
        if self.current_path and os.path.isfile(self.current_path):
            current_norm = os.path.normcase(os.path.abspath(self.current_path))
            if all(os.path.normcase(os.path.abspath(path)) != current_norm for path in paths):
                paths.insert(0, self.current_path)

        self._trace_review_perf(
            "load_pieces_discovered_paths",
            trace_started,
            force=True,
            stream=bool(stream_sidebar),
            paths=len(paths),
        )
        metadata_started = time.perf_counter()
        progress_map = self._read_progress_metadata()
        paths = self._deduplicate_review_sidecar_paths(paths, progress_map)
        spine_positions = self._read_spine_positions(allow_deep_search=not stream_sidebar)
        work_items = []
        for fallback_index, path in enumerate(paths):
            metadata = self._sidecar_metadata(path, fallback_index, progress_map, spine_positions)
            work_items.append((path, metadata))
        work_items.sort(key=lambda item: item[1].get("sort_key", (999999, 999999, "")))

        display_numbers = nonreset_chapter_display_numbers(
            metadata.get("chapter_num") for _path, metadata in work_items
        )
        for index, ((_path, metadata), display_chapter_num) in enumerate(
            zip(work_items, display_numbers)
        ):
            if metadata.get("opf_position") is None:
                metadata["display_position"] = index + 1
            else:
                metadata["display_position"] = int(metadata["opf_position"]) + 1
            metadata["raw_chapter_num"] = metadata.get("chapter_num")
            metadata["display_chapter_num"] = display_chapter_num
            metadata["label"] = self._review_label_from_metadata(metadata)
        self._trace_review_perf(
            "load_pieces_metadata_ready",
            metadata_started,
            force=True,
            stream=bool(stream_sidebar),
            work_items=len(work_items),
        )

        stream_sidebar = bool(stream_sidebar and self.isVisible())
        if stream_sidebar:
            stream_sidebar = self._prepare_streaming_piece_list(work_items)
            if stream_sidebar:
                try:
                    self._streaming_piece_total = len(work_items)
                    self.save_status_label.setText(f"Loading SDLXLIFF entries 0/{len(work_items)}")
                    self._set_loading_progress(0, len(work_items), f"0/{len(work_items)} SDLXLIFF entries")
                except Exception:
                    pass

        def flush_streamed_pieces(limit=None):
            if not stream_sidebar:
                return 0
            flushed = 0
            nonlocal next_stream_index
            while next_stream_index < len(pieces) and pieces[next_stream_index] is not None:
                self._stream_piece_list_item(next_stream_index, pieces[next_stream_index])
                next_stream_index += 1
                flushed += 1
                if limit is not None and flushed >= limit:
                    break
            if flushed:
                self._pump_review_loading_events(max_ms=4)
            return flushed

        pieces = []
        next_stream_index = 0
        if len(work_items) <= 1:
            pieces = [
                self._build_piece(path, idx, metadata)
                for idx, (path, metadata) in enumerate(work_items)
            ]
            if stream_sidebar:
                flush_streamed_pieces()
                self._finish_streaming_piece_list()
                self._trace_review_perf(
                    "load_pieces_done",
                    trace_started,
                    force=True,
                    stream=True,
                    pieces=len(self.pieces or []),
                    work_items=len(work_items),
                )
                return list(self.pieces)
            filtered = self._filter_review_pieces(pieces)
            self._trace_review_perf(
                "load_pieces_done",
                trace_started,
                force=True,
                stream=False,
                pieces=len(filtered),
                work_items=len(work_items),
            )
            return filtered

        pieces = [None] * len(work_items)
        max_workers = self._review_piece_worker_count(len(work_items))

        # Single-worker streaming load: parse in one background worker while
        # the GUI thread flushes finished entries into the sidebar.
        # Use the Other Settings parallel-extraction worker count; a value of
        # 1 keeps the old single background parser behavior.
        if stream_sidebar and max_workers <= 1:
            stream_started = time.perf_counter()
            total = len(work_items)
            worker_done = threading.Event()

            def _parse_worker():
                try:
                    for idx, (path, metadata) in enumerate(work_items):
                        pieces[idx] = self._build_piece(path, idx, metadata)
                finally:
                    worker_done.set()

            worker = threading.Thread(target=_parse_worker, name="sdlxliff-piece-parse", daemon=True)
            worker.start()
            while True:
                flushed = flush_streamed_pieces(limit=16)
                if worker_done.is_set() and next_stream_index >= total:
                    break
                self._pump_review_loading_events(max_ms=8)
                if not flushed:
                    time.sleep(0.01)
            self._finish_streaming_piece_list()
            self._trace_review_perf(
                "load_pieces_stream_done",
                stream_started,
                force=True,
                pieces=len(self.pieces or []),
                work_items=total,
                workers=max_workers,
            )
            self._trace_review_perf(
                "load_pieces_done",
                trace_started,
                force=True,
                stream=True,
                pieces=len(self.pieces or []),
                work_items=total,
            )
            return list(self.pieces)
        parallel_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._build_piece, path, idx, metadata): idx
                for idx, (path, metadata) in enumerate(work_items)
            }
            pending = set(futures)
            while pending or (stream_sidebar and next_stream_index < len(pieces)):
                if stream_sidebar and flush_streamed_pieces(limit=12):
                    continue
                if not pending:
                    break
                done, pending = wait(pending, timeout=0.025, return_when=FIRST_COMPLETED)
                if not done:
                    if stream_sidebar:
                        self._pump_review_loading_events(max_ms=4)
                    continue
                for future in done:
                    idx = futures[future]
                    try:
                        pieces[idx] = future.result()
                    except Exception as exc:
                        path, metadata = work_items[idx]
                        pieces[idx] = self._build_piece(path, idx, metadata)
                        pieces[idx]["error"] = str(exc)
                flush_streamed_pieces(limit=12)
                if stream_sidebar:
                    self._pump_review_loading_events(max_ms=4)
        if stream_sidebar:
            flush_streamed_pieces()
            self._finish_streaming_piece_list()
            self._trace_review_perf(
                "load_pieces_stream_done",
                parallel_started,
                force=True,
                pieces=len(self.pieces or []),
                work_items=len(work_items),
                workers=max_workers,
            )
            self._trace_review_perf(
                "load_pieces_done",
                trace_started,
                force=True,
                stream=True,
                pieces=len(self.pieces or []),
                work_items=len(work_items),
            )
            return list(self.pieces)
        self._trace_review_perf(
            "load_pieces_parallel_done",
            parallel_started,
            force=True,
            pieces=len([piece for piece in pieces if piece is not None]),
            work_items=len(work_items),
            workers=max_workers,
        )
        filtered = self._filter_review_pieces(pieces)
        self._trace_review_perf(
            "load_pieces_done",
            trace_started,
            force=True,
            stream=False,
            pieces=len(filtered),
            work_items=len(work_items),
        )
        return filtered

    def _row_for_piece_path(self, path):
        if not path:
            return None
        try:
            target_norm = os.path.normcase(os.path.abspath(path))
        except Exception:
            return None
        for row, piece in enumerate(self.pieces):
            try:
                piece_norm = os.path.normcase(os.path.abspath(piece.get("path") or ""))
            except Exception:
                continue
            if piece_norm == target_norm:
                return row
        return None

    def _review_context_menu_is_open(self):
        return bool(getattr(self, "_review_context_menu_open", False))

    @staticmethod
    def _review_row_snapshot(row_data):
        row_data = row_data if isinstance(row_data, dict) else {}
        return {
            "source": str(row_data.get("source", "") or ""),
            "target": str(row_data.get("target", "") or ""),
            "source_tag": str(row_data.get("source_tag", "") or ""),
            "target_tag": str(row_data.get("target_tag", "") or ""),
            "source_missing": bool(row_data.get(
                "source_missing", not row_data.get("source_tag")
            )),
            "target_missing": bool(row_data.get(
                "target_missing", not row_data.get("target_tag")
            )),
            "translator_note": bool(row_data.get("translator_note")),
            "status": str(row_data.get("status", "green") or "green"),
            "tooltip_translation": str(row_data.get("tooltip_translation", "") or ""),
            "tooltip_translation_pending": bool(row_data.get("tooltip_translation_pending")),
            "tooltip_translation_status": str(row_data.get("tooltip_translation_status", "") or ""),
            "tooltip_translation_error": str(row_data.get("tooltip_translation_error", "") or ""),
            "tooltip_translation_error_detail": str(row_data.get("tooltip_translation_error_detail", "") or ""),
        }

    @classmethod
    def _compact_review_row_visible(cls, row_data):
        """Hide only empty translator-note placeholders from Compact mode."""
        row_data = row_data if isinstance(row_data, dict) else {}
        return not (
            bool(row_data.get("translator_note"))
            and not cls._normalize_review_text(row_data.get("target", ""))
        )

    @classmethod
    def _compact_translator_note_display_label(cls, piece, row_index):
        """Number only visible TN cards without changing persisted ordinals."""
        visible_ordinal = 0
        for index, row_data in enumerate((piece or {}).get("rows") or []):
            if index > int(row_index):
                break
            if not row_data.get("translator_note"):
                continue
            if not cls._compact_review_row_visible(row_data):
                continue
            visible_ordinal += 1
            if index == int(row_index):
                return f"TN({visible_ordinal})"
        return ""

    @staticmethod
    def _review_wrapped_lines(value, line_chars):
        text = str(value or "")
        if not text:
            return 1
        line_chars = max(1, int(line_chars or 1))
        wrapped_lines = 0
        for part in text.splitlines() or [text]:
            wrapped_lines += max(1, (len(part) + line_chars - 1) // line_chars)
        return wrapped_lines

    @classmethod
    def _review_chars_per_line_for_width(cls, viewport_width=1200, two_column_layout=False):
        viewport_width = max(700, int(viewport_width or 1200))
        fixed_width = 92 + 180 + 26 + 180 + 20 + 50
        if two_column_layout:
            fixed_width = 92 + 250 + 46
        text_column_width = max(180, (viewport_width - fixed_width) // 2)
        if two_column_layout:
            text_column_width = max(320, viewport_width - fixed_width)
        return max(24, int(text_column_width / 9))

    @classmethod
    def _review_row_line_counts_for_width(
        cls,
        source_text,
        target_text,
        tooltip_translation=None,
        tooltip_pending=False,
        viewport_width=1200,
        two_column_layout=False,
        tooltip_preview_text=None,
    ):
        chars_per_line = cls._review_chars_per_line_for_width(
            viewport_width,
            two_column_layout=two_column_layout,
        )
        source_lines = cls._review_wrapped_lines(source_text, chars_per_line)
        target_lines = cls._review_wrapped_lines(target_text, chars_per_line)
        tooltip_preview = (
            str(tooltip_preview_text or "").strip()
            if tooltip_preview_text is not None
            else (cls.MACHINE_TRANSLATION_PENDING_TEXT if tooltip_pending else str(tooltip_translation or "").strip())
        )
        if tooltip_preview:
            translated_chars_per_line = max(30, int(chars_per_line * 1.35))
            tooltip_lines = min(18, cls._review_wrapped_lines(tooltip_preview, translated_chars_per_line))
        else:
            tooltip_lines = 0
        return source_lines, target_lines, tooltip_lines

    @classmethod
    def _review_row_height_for_width(
        cls,
        source_text,
        target_text,
        tooltip_translation=None,
        tooltip_pending=False,
        viewport_width=1200,
        two_column_layout=False,
        tooltip_preview_text=None,
    ):
        source_lines, target_lines, tooltip_lines = cls._review_row_line_counts_for_width(
            source_text,
            target_text,
            tooltip_translation,
            tooltip_pending,
            viewport_width,
            two_column_layout=two_column_layout,
            tooltip_preview_text=tooltip_preview_text,
        )
        max_lines = max(1, source_lines, target_lines)
        tooltip_preview = (
            str(tooltip_preview_text or "").strip()
            if tooltip_preview_text is not None
            else (cls.MACHINE_TRANSLATION_PENDING_TEXT if tooltip_pending else str(tooltip_translation or "").strip())
        )
        if tooltip_lines:
            max_lines = max(max_lines, source_lines + tooltip_lines)
        if two_column_layout:
            source_height = max(34, (source_lines + tooltip_lines) * 22 + (28 if tooltip_lines else 10))
            target_height = max(cls.REVIEW_TARGET_EDIT_MIN_HEIGHT, target_lines * 22 + 28)
            text_height = 10 + source_height + 7 + target_height
            controls_height = 3 * 34 + 2 * 5 + 24
            height = min(
                cls.REVIEW_ROW_MAX_HEIGHT,
                max(cls.REVIEW_ROW_MIN_HEIGHT, text_height, controls_height),
            )
            if tooltip_preview:
                height = min(cls.REVIEW_ROW_MAX_HEIGHT, max(height, cls.REVIEW_ROW_MIN_HEIGHT + 40))
            return height
        extra_lines = max(0, min(12, max_lines - 1))
        height = min(cls.REVIEW_ROW_MAX_HEIGHT, cls.REVIEW_ROW_MIN_HEIGHT + extra_lines * 22)
        if tooltip_preview:
            height = min(cls.REVIEW_ROW_MAX_HEIGHT, max(height, cls.REVIEW_ROW_MIN_HEIGHT + 30))
        return height

    @classmethod
    def _build_review_piece_render_model_from_rows(cls, row_snapshots, viewport_width, two_column_layout=False):
        rows = list(row_snapshots or [])
        max_len = max(
            [len(row.get("source", "")) for row in rows]
            + [len(row.get("target", "")) for row in rows]
            + [1]
        )
        row_models = []
        for row in rows:
            source_text = row.get("source", "")
            target_text = row.get("target", "")
            tooltip_translation = row.get("tooltip_translation", "")
            tooltip_pending = bool(row.get("tooltip_translation_pending"))
            tooltip_preview = cls._row_machine_translation_preview_from_snapshot(row)
            tooltip_state = cls._row_machine_translation_preview_state(row)
            source_lines, target_lines, tooltip_lines = cls._review_row_line_counts_for_width(
                source_text,
                target_text,
                tooltip_translation,
                tooltip_pending,
                viewport_width,
                two_column_layout=two_column_layout,
                tooltip_preview_text=tooltip_preview,
            )
            source_missing = bool(row.get(
                "source_missing", not row.get("source_tag")
            ))
            target_missing = bool(row.get(
                "target_missing", not row.get("target_tag")
            ))
            row_models.append({
                "source_text": source_text,
                "target_text": target_text,
                "source_len": len(source_text),
                "target_len": len(target_text),
                "source_missing": source_missing,
                "target_missing": target_missing,
                "target_editable": (not source_missing or not target_missing),
                "translator_note": bool(row.get("translator_note")),
                "tooltip_translation": tooltip_translation,
                "tooltip_pending": tooltip_pending,
                "tooltip_preview": tooltip_preview,
                "tooltip_state": tooltip_state,
                "tooltip_detail": str(row.get("tooltip_translation_error_detail") or row.get("tooltip_translation_status") or ""),
                "two_column_layout": bool(two_column_layout),
                "source_lines": source_lines,
                "target_lines": target_lines,
                "tooltip_lines": tooltip_lines,
                "row_height": cls._review_row_height_for_width(
                    source_text,
                    target_text,
                    tooltip_translation,
                    tooltip_pending,
                    viewport_width,
                    two_column_layout=two_column_layout,
                    tooltip_preview_text=tooltip_preview,
                ),
            })
        return {
            "viewport_width": int(max(700, viewport_width or 1200)),
            "two_column_layout": bool(two_column_layout),
            "row_count": len(rows),
            "max_len": max_len,
            "rows": row_models,
        }

    @classmethod
    def _build_notepad_review_rows(cls, piece):
        """Map every target HTML element to one continuous notepad line."""
        try:
            from bs4 import BeautifulSoup
        except Exception:
            return list(piece.get("rows") or [])

        review_rows = list(piece.get("rows") or [])
        by_target_index = {
            row.get("target_index"): (row_index, row)
            for row_index, row in enumerate(review_rows)
            if isinstance(row.get("target_index"), int)
        }
        html_text = cls._unescape_html_document(piece.get("target_html") or "")
        if not str(html_text or "").strip():
            html_text = cls._unescape_html_document(piece.get("source_html") or "")
        soup = BeautifulSoup(html_text, "html.parser")
        text_tags = set(cls.TEXT_TAGS)
        consumed_rows = set()
        notepad_rows = []
        text_unit_index = 0

        def _is_text_unit(tag):
            name = str(getattr(tag, "name", "") or "").casefold()
            if name in text_tags:
                return True
            if name != "div":
                return False
            classes = tag.get("class") or []
            if isinstance(classes, str):
                classes = classes.split()
            return "u" in {str(value).casefold() for value in classes} and not tag.find(cls.TEXT_TAGS)

        def _tag_caption(tag, depth):
            name = str(getattr(tag, "name", "") or "").casefold() or "?"
            attributes = []
            for key, value in list((getattr(tag, "attrs", {}) or {}).items()):
                if isinstance(value, (list, tuple)):
                    value = " ".join(str(item) for item in value)
                if value in (None, ""):
                    attributes.append(str(key))
                else:
                    attributes.append(f'{key}="{value}"')
            suffix = (" " + " ".join(attributes)) if attributes else ""
            if len(suffix) > 100:
                suffix = suffix[:97] + "..."
            is_empty = not tag.find(True, recursive=False) and not tag.get_text(" ", strip=True)
            void = name in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
            close = " />" if void or is_empty else f"> … </{name}>"
            return f'{"  " * min(max(0, depth), 10)}<{name}{suffix}{close}', is_empty or void

        for dom_order, tag in enumerate(soup.find_all(True)):
            depth = sum(
                1 for parent in getattr(tag, "parents", [])
                if str(getattr(parent, "name", "") or "") not in {"", "[document]"}
            )
            caption, is_empty = _tag_caption(tag, depth)
            direct_text = cls._normalize_review_text(
                " ".join(
                    str(child).strip()
                    for child in list(getattr(tag, "contents", []) or [])
                    if getattr(child, "name", None) is None and str(child).strip()
                )
            )
            match = None
            if _is_text_unit(tag):
                match = by_target_index.get(text_unit_index)
                text_unit_index += 1
            if match is not None:
                row_index, review_row = match
                item = dict(review_row)
                item.update({
                    "notepad_dom_order": dom_order,
                    "notepad_tag_caption": caption,
                    "notepad_depth": depth,
                    "notepad_empty": is_empty,
                    "notepad_inline_text": direct_text,
                    "notepad_structural": False,
                    "review_row_index": row_index,
                })
                consumed_rows.add(row_index)
            else:
                item = {
                    "source": "",
                    "target": "",
                    "source_tag": "",
                    "target_tag": str(getattr(tag, "name", "") or "").casefold(),
                    "status": "structural",
                    "reason": "Empty or structural HTML element",
                    "notepad_dom_order": dom_order,
                    "notepad_tag_caption": caption,
                    "notepad_depth": depth,
                    "notepad_empty": is_empty,
                    "notepad_inline_text": direct_text,
                    "notepad_structural": True,
                    "review_row_index": -1,
                }
            notepad_rows.append(item)

        # A malformed/incomplete target can omit a source unit entirely. Keep
        # it editable after the parsed target sequence instead of hiding it.
        for row_index, review_row in enumerate(review_rows):
            if row_index in consumed_rows:
                continue
            item = dict(review_row)
            tag_name = review_row.get("source_tag") or review_row.get("target_tag") or "p"
            item.update({
                "notepad_dom_order": len(notepad_rows),
                "notepad_tag_caption": f"<{tag_name}> … </{tag_name}>",
                "notepad_depth": 0,
                "notepad_empty": not bool(review_row.get("target")),
                "notepad_inline_text": "",
                "notepad_structural": False,
                "review_row_index": row_index,
            })
            notepad_rows.append(item)
        return notepad_rows

    def _review_rows_for_current_layout(self, piece):
        if bool(getattr(self, "_two_column_layout_enabled", True)):
            return piece.get("rows") or []
        cached = piece.get("_notepad_rows") if isinstance(piece, dict) else None
        if not isinstance(cached, list):
            cached = self._build_notepad_review_rows(piece)
            piece["_notepad_rows"] = cached
        return cached

    def _piece_render_snapshot(self, piece, rows=None):
        return [
            self._review_row_snapshot(row_data)
            for row_data in (rows if rows is not None else (piece.get("rows") or []))
        ]

    def _invalidate_piece_render_model(self, piece=None, restart_preload=True):
        try:
            if isinstance(piece, dict):
                piece.pop("_render_model", None)
                piece.pop("_notepad_rows", None)
            self._review_data_preload_token = int(getattr(self, "_review_data_preload_token", 0)) + 1
            if restart_preload and getattr(self, "_review_data_loaded", False):
                self._queue_review_data_preload(delay_ms=250)
        except Exception:
            pass

    def _review_selection_recently_changed(self):
        try:
            return (time.monotonic() - float(self._last_review_selection_change or 0.0)) < (
                self.REVIEW_PRELOAD_IDLE_MS / 1000.0
            )
        except Exception:
            return False

    def _review_scroll_recently_active(self, threshold_s=0.25):
        try:
            last = float(getattr(self, "_last_review_scroll_activity", 0.0) or 0.0)
            return (time.monotonic() - last) < float(threshold_s)
        except Exception:
            return False

    def _review_preload_order(self, current_row):
        if not self.pieces:
            return []
        try:
            current_row = int(current_row)
        except Exception:
            current_row = 0
        rows = []
        for distance in range(1, self.REVIEW_PRELOAD_RADIUS + 1):
            for row in (current_row + distance, current_row - distance):
                if (
                    0 <= row < len(self.pieces)
                    and row not in self._piece_render_complete
                    and row not in self._piece_pages
                    and row not in rows
                ):
                    rows.append(row)
        return rows

    @staticmethod
    def _tag_label_text(source_tag, target_tag, source_label=None, target_label=None):
        source_tag = str(source_tag or "").strip()
        target_tag = str(target_tag or "").strip()
        source_label = str(source_label or source_tag).strip()
        target_label = str(target_label or target_tag).strip()
        source_ordinal = re.search(r"\((\d+)\)", source_label)
        target_ordinal = re.search(r"\((\d+)\)", target_label)
        if not source_tag and not target_tag:
            for label in (target_label, source_label):
                translator_note = re.fullmatch(
                    r"tn\((\d+)\)", label, flags=re.IGNORECASE
                )
                if translator_note:
                    return f"TN({translator_note.group(1)})"
        if source_tag and target_tag:
            return source_label if source_label == target_label else f"{source_label} -> {target_label}"
        elif source_tag:
            return f"Empty({source_ordinal.group(1)})" if source_ordinal else "Empty"
        elif target_tag:
            return f"Added({target_ordinal.group(1)})" if target_ordinal else f"+ {target_label}"
        return "-"

    @staticmethod
    def _tag_label_ordinal_font_point_size(font_point_size):
        return max(3.0, min(8.0, float(font_point_size) - 2.0))

    @classmethod
    def _tag_label_rich_text(cls, text, font_point_size=11.0):
        escaped = html_lib.escape(str(text or ""))
        ordinal_point_size = cls._tag_label_ordinal_font_point_size(
            font_point_size
        )
        return re.sub(
            r"\((\d+)\)",
            lambda match: (
                f'<span style="font-size: {ordinal_point_size:g}pt;">'
                f'({match.group(1)})</span>'
            ),
            escaped,
        )

    @staticmethod
    def _wrapped_tooltip(text, width=560):
        text = str(text or "").strip()
        if not text:
            return ""
        escaped = html_lib.escape(text).replace("\n", "<br>")
        return f'<div style="white-space: normal; width: {int(width)}px;">{escaped}</div>'

    def _review_status_colors(self):
        return {
            "green": (self.THEME["panel"], self.THEME["accent"], self.THEME["info"], self.THEME["success"], self.THEME["border"]),
            "yellow": ("#3d3320", self.THEME["accent"], self.THEME["warning"], self.THEME["warning"], self.THEME["warning"]),
            "purple": ("#32243f", self.THEME["accent"], self.THEME["purple"], self.THEME["purple"], self.THEME["purple"]),
            "red": ("#3a2428", self.THEME["accent"], self.THEME["danger"], self.THEME["danger"], self.THEME["danger"]),
        }

    def _inject_machine_translation_to_target(self, piece_index, row_index, translated, editor=None):
        translated = str(translated or "").strip()
        if not translated:
            return
        if self._insert_into_review_editor(editor, translated):
            return
        self._apply_target_edit(piece_index, row_index, translated)

    def _undo_all_target_edits(self, piece_index, row_index, editor=None):
        original = ""
        try:
            if 0 <= piece_index < len(self.pieces):
                rows = self.pieces[piece_index].get("rows") or []
                if 0 <= row_index < len(rows):
                    original = str(rows[row_index].get("target_original", "") or "")
        except Exception:
            original = ""
        if self._insert_into_review_editor(editor, original):
            return
        self._apply_target_edit(piece_index, row_index, original)

    def _review_target_language(self):
        language = ""
        try:
            language = str((self._config or {}).get("output_language") or "").strip()
        except Exception:
            language = ""
        return language or "English"

    def _review_target_language_code(self):
        return self._GT_LANG_CODES.get(self._review_target_language().lower(), "en")

    @staticmethod
    def _machine_translation_source_hash(text):
        return hashlib.sha256(str(text or "").encode("utf-8", errors="replace")).hexdigest()

    def _machine_translation_row_key(self, row_data, target_code=None):
        target_code = target_code or self._review_target_language_code()
        source_index = row_data.get("source_index")
        if source_index is None:
            source_index = row_data.get("row_index")
        source_tag = str(row_data.get("source_tag", "") or "").strip().lower()
        source_hash = self._machine_translation_source_hash(row_data.get("source", ""))
        return f"{target_code}|{source_index}|{source_tag}|{source_hash}"

    def _machine_translation_path_for_piece(self, piece):
        return _sdlxliff_machine_translation_path(
            getattr(self, "output_dir", "") or "",
            piece.get("path") or piece.get("output_name") or piece.get("name"),
        )

    def _read_machine_translation_file(self, piece):
        path = self._machine_translation_path_for_piece(piece)
        if not path or not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries") if isinstance(data, dict) else {}
            return entries if isinstance(entries, dict) else {}
        except Exception:
            return {}

    def _load_machine_translation_file_for_piece(self, piece):
        rows = piece.get("rows") or []
        if not rows:
            return
        entries = self._read_machine_translation_file(piece)
        if not entries:
            return
        target_code = self._review_target_language_code()
        for row_data in rows:
            key = self._machine_translation_row_key(row_data, target_code)
            entry = entries.get(key)
            if not isinstance(entry, dict):
                continue
            if entry.get("target_language") != target_code:
                continue
            if entry.get("source_hash") != self._machine_translation_source_hash(row_data.get("source", "")):
                continue
            translated = str(entry.get("translation") or "").strip()
            if not translated:
                continue
            row_data["tooltip_translation"] = translated

    def _reload_machine_translation_previews(self, signature=None):
        changed_by_piece = {}
        try:
            for piece_index, piece in enumerate(self.pieces or []):
                rows = piece.get("rows") or []
                if not rows:
                    continue
                before = [
                    (
                        str(row_data.get("tooltip_translation") or ""),
                        str(row_data.get("status") or ""),
                        str(row_data.get("reason") or ""),
                    )
                    for row_data in rows
                ]
                for row_data in rows:
                    row_data.pop("tooltip_translation", None)
                self._load_machine_translation_file_for_piece(piece)
                self._refresh_piece_summary(piece)
                changed_rows = []
                for row_index, row_data in enumerate(rows):
                    after = (
                        str(row_data.get("tooltip_translation") or ""),
                        str(row_data.get("status") or ""),
                        str(row_data.get("reason") or ""),
                    )
                    if before[row_index] != after:
                        row_data["_source_preview_dirty"] = True
                        changed_rows.append(row_index)
                if changed_rows:
                    self._invalidate_piece_render_model(piece, restart_preload=False)
                    self._refresh_piece_list_item(piece_index)
                    changed_by_piece[piece_index] = changed_rows

            current_row = self._displayed_piece_row()
            for piece_index, changed_rows in changed_by_piece.items():
                if piece_index == current_row:
                    self._refresh_piece_header(piece_index)
                    for row_index in changed_rows:
                        self._refresh_visible_review_row_status(piece_index, row_index)
                if piece_index == current_row and not self._review_context_menu_is_open():
                    self._update_review_row_source_previews(piece_index, changed_rows, visible_only=True)
                elif piece_index == current_row:
                    self._queue_refresh_current_visible_dirty_source_previews()
        except Exception:
            pass
        try:
            self._last_machine_translation_signature = (
                signature if signature is not None else self._current_machine_translation_signature()
            )
        except Exception:
            pass

    def _write_machine_translation_entries(self, piece, row_translations):
        row_translations = [
            (row_data, str(translated or "").strip())
            for row_data, translated in (row_translations or [])
            if str(translated or "").strip()
        ]
        if not row_translations:
            return
        path = self._machine_translation_path_for_piece(piece)
        if not path:
            return
        target_code = self._review_target_language_code()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {}
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        data = loaded
                except Exception:
                    data = {}
            entries = data.get("entries")
            if not isinstance(entries, dict):
                entries = {}

            for row_data, translated in row_translations:
                entry_key = self._machine_translation_row_key(row_data, target_code)
                source_index = row_data.get("source_index")
                if source_index is None:
                    source_index = row_data.get("row_index")
                source_tag = str(row_data.get("source_tag", "") or "").strip().lower()
                source_hash = self._machine_translation_source_hash(row_data.get("source", ""))
                entries[entry_key] = {
                    "source_index": source_index,
                    "source_tag": source_tag,
                    "source_hash": source_hash,
                    "target_language": target_code,
                    "translation": translated,
                }

            data.update({
                "version": 1,
                "sidecar": os.path.basename(str(piece.get("path") or piece.get("output_name") or "")),
                "entries": entries,
            })
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _write_machine_translation_entry(self, piece, row_data, translated):
        translated = str(translated or "").strip()
        if not translated:
            return
        self._write_machine_translation_entries(piece, [(row_data, translated)])

    def _tooltip_translation_key(self, piece, row_data):
        piece_path = ""
        try:
            piece_path = os.path.normcase(os.path.abspath(piece.get("path") or piece.get("output_name") or ""))
        except Exception:
            piece_path = str(piece.get("path") or piece.get("output_name") or "")
        source_index = row_data.get("source_index")
        if source_index is None:
            source_index = row_data.get("row_index")
        return (
            piece_path,
            self._review_target_language_code(),
            source_index,
            str(row_data.get("source_tag", "") or "").strip().lower(),
            self._machine_translation_source_hash(row_data.get("source", "")),
        )

    def _row_tooltip_translation(self, piece, row_data):
        stored = row_data.get("tooltip_translation")
        if stored:
            return str(stored)
        return ""

    def _set_row_tooltip_translation(self, piece, row_data, translated, persist=True):
        translated = str(translated or "").strip()
        if not translated:
            return
        try:
            from google_free_translate import GoogleFreeTranslateNew
            translated = GoogleFreeTranslateNew._sanitize_argos_text_tag_fragments(row_data.get("source", ""), translated)
        except Exception:
            pass
        translated = str(translated or "").strip()
        if not translated:
            return
        if self._normalized_machine_translation_text(translated) == self._normalized_machine_translation_text(row_data.get("source", "")):
            return
        row_data["tooltip_translation"] = translated
        row_data["_source_preview_dirty"] = True
        self._invalidate_piece_render_model(piece, restart_preload=False)
        if persist:
            self._write_machine_translation_entry(piece, row_data, translated)

    @staticmethod
    def _tooltip_batch_tag_name(tag_name):
        tag_name = str(tag_name or "").strip().lower()
        if re.fullmatch(r"h[1-6]", tag_name) or tag_name in {"p", "li"}:
            return tag_name
        return "p"

    def _tooltip_batch_html(self, work):
        parts = []
        for pos, (_row_idx, _key, source_text, tag_name) in enumerate(work):
            tag_name = self._tooltip_batch_tag_name(tag_name)
            escaped = html_lib.escape(str(source_text or ""), quote=False)
            parts.append(f'<{tag_name} data-sdl-tip="{pos}">{escaped}</{tag_name}>')
        return "\n".join(parts)

    def _extract_tooltip_batch_translations(self, translated_html, work):
        translated_html = self._unescape_html_document(translated_html)
        if not translated_html.strip() or not work:
            return {}
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(translated_html, "html.parser")
            nodes = [
                node for node in soup.find_all(self.TEXT_TAGS)
                if node.get_text(" ", strip=True)
            ]
            by_position = {}
            claimed = set()
            for node in nodes:
                raw_pos = node.get("data-sdl-tip")
                if raw_pos is None:
                    raw_pos = node.get("data-sdl-tip".lower())
                try:
                    pos = int(raw_pos)
                except (TypeError, ValueError):
                    continue
                if 0 <= pos < len(work):
                    by_position[pos] = node.get_text(" ", strip=True)
                    claimed.add(id(node))
            if len(by_position) < len(work):
                # Raw-markup recovery: providers often return literal angle
                # brackets instead of entities (e.g. <Prologue> for
                # &lt;Prologue&gt;). The parser then treats that as a tag,
                # the wrapper node's text parses empty, and the unit is
                # lost. Re-extract the wrapper's inner content straight
                # from the UNPARSED response and treat it as literal text.
                for pos in range(len(work)):
                    if pos in by_position:
                        continue
                    m = re.search(
                        r'<([a-zA-Z][\w-]*)\b[^>]*data-sdl-tip\s*=\s*["\']'
                        + str(pos)
                        + r'["\'][^>]*>(.*?)</\1\s*>',
                        translated_html,
                        re.S | re.I,
                    )
                    if not m:
                        continue
                    inner = m.group(2)
                    # Drop well-formed formatting tags but keep unknown
                    # angle-bracket content (it's the translation itself).
                    inner = re.sub(
                        r'</?(?:b|i|em|strong|span|u|small|sub|sup|br|a|font)\b[^>]*/?>',
                        '', inner, flags=re.I)
                    text = self._normalize_review_text(re.sub(r'\s+', ' ', inner))
                    if text:
                        by_position[pos] = text
            if len(by_position) < len(work):
                # Fill ONLY the missing positions, and ONLY from nodes that
                # didn't claim a position via data-sdl-tip. The old fallback
                # re-mapped ALL node texts onto positions 0..N in order, so
                # when the provider mangled one unit (e.g. returned literal
                # <Prologue>, whose node parses empty and gets dropped),
                # every translation shifted and row 0 displayed row 1's
                # text. With aligned claims, a mangled unit now just stays
                # blank instead of stealing its neighbor's translation.
                unclaimed_texts = [
                    node.get_text(" ", strip=True)
                    for node in nodes
                    if id(node) not in claimed and node.get_text(" ", strip=True)
                ]
                missing_positions = [
                    pos for pos in range(len(work)) if pos not in by_position
                ]
                for pos, text in zip(missing_positions, unclaimed_texts):
                    by_position[pos] = text
            return {
                key: by_position[pos]
                for pos, (_row_idx, key, _source_text, _tag_name) in enumerate(work)
                if str(by_position.get(pos, "")).strip()
            }
        except Exception:
            return {}

    @staticmethod
    def _normalized_machine_translation_text(text):
        text = unicodedata.normalize("NFKC", html_lib.unescape(str(text or ""))).strip().lower()
        return re.sub(r"\s+", " ", text)

    @classmethod
    def _compact_machine_translation_text(cls, text):
        return re.sub(r"\s+", "", cls._normalized_machine_translation_text(text))

    def _validate_tooltip_batch_translations(self, translations, work):
        if not translations:
            return {}, "Machine translation returned no parseable preview translations."
        source_by_key = {
            key: str(source_text or "")
            for _row_idx, key, source_text, _tag_name in (work or [])
        }
        valid = {}
        unchanged = []
        for key, translated in (translations or {}).items():
            translated_text = str(translated or "").strip()
            source_text = source_by_key.get(key, "")
            if not translated_text:
                continue
            if self._normalized_machine_translation_text(translated_text) == self._normalized_machine_translation_text(source_text):
                unchanged.append(key)
                continue
            valid[key] = translated_text
        if not valid:
            if unchanged:
                return {}, (
                    "Machine translation returned source text unchanged for "
                    f"{len(unchanged)}/{len(work or [])} row(s); refusing to save raw source as preview."
                )
            return {}, "Machine translation returned no usable preview translations."
        if unchanged:
            return valid, (
                "Machine translation returned source text unchanged for "
                f"{len(unchanged)}/{len(work or [])} row(s); skipped those rows."
            )
        return valid, ""

    def _mark_review_sidecars_completed(self, piece_rows):
        rows = sorted({row for row in (piece_rows or []) if 0 <= row < len(self.pieces)})
        if not rows:
            return
        marked = 0
        errors = []
        current_row = self._displayed_piece_row()
        for piece_index in rows:
            try:
                piece = self.pieces[piece_index]
            except Exception:
                continue
            if not self._piece_needs_manual_green_override(piece):
                continue
            progress_result = self._mark_piece_progress_completed(piece)
            if not progress_result.get("ok"):
                errors.append(str(progress_result.get("error") or "progress update failed"))
                continue
            before = [
                (str(row_data.get("status") or ""), str(row_data.get("reason") or ""))
                for row_data in (piece.get("rows") or [])
            ]
            self._apply_manual_green_override_to_piece(piece)
            if not self._persist_piece_manual_green_override(piece):
                restored = self._restore_piece_progress_before_manual_completion(piece)
                if restored:
                    self._clear_piece_manual_green_override(piece, persist=False)
                errors.append("could not save the completed review mark")
                continue
            marked += 1

            changed_rows = [
                row_index for row_index, row_data in enumerate(piece.get("rows") or [])
                if row_index >= len(before)
                or before[row_index] != (str(row_data.get("status") or ""), str(row_data.get("reason") or ""))
            ]
            self._invalidate_piece_render_model(piece, restart_preload=False)
            self._refresh_piece_list_item(piece_index)
            if piece_index == current_row:
                self._refresh_piece_header(piece_index)
                for row_index in changed_rows:
                    self._refresh_visible_review_row_status(piece_index, row_index)
            else:
                self._invalidate_piece_page_for_refresh(piece_index)

        try:
            if marked <= 0:
                if errors:
                    self.save_status_label.setText(f"Could not mark completed: {errors[0]}")
                else:
                    self.save_status_label.setText("No eligible SDLXLIFF sidecars selected")
            elif errors:
                self.save_status_label.setText(
                    f"Marked {marked} completed; {len(errors)} progress update{'s' if len(errors) != 1 else ''} failed"
                )
            else:
                self.save_status_label.setText(
                    f"Marked {marked} SDLXLIFF sidecar{'s' if marked != 1 else ''} completed"
                )
        except Exception:
            pass

    def _undo_review_sidecars_completed(self, piece_rows):
        rows = sorted({row for row in (piece_rows or []) if 0 <= row < len(self.pieces)})
        if not rows:
            return
        undone = 0
        current_row = self._displayed_piece_row()
        for piece_index in rows:
            try:
                piece = self.pieces[piece_index]
            except Exception:
                continue
            if not piece.get("manual_green_override"):
                continue
            before = [
                (str(row_data.get("status") or ""), str(row_data.get("reason") or ""))
                for row_data in (piece.get("rows") or [])
            ]
            if not self._clear_piece_manual_green_override(piece, persist=True):
                continue
            self._recompute_piece_row_statuses(piece)
            self._refresh_piece_summary(piece)
            undone += 1

            changed_rows = [
                row_index for row_index, row_data in enumerate(piece.get("rows") or [])
                if row_index >= len(before)
                or before[row_index] != (str(row_data.get("status") or ""), str(row_data.get("reason") or ""))
            ]
            self._invalidate_piece_render_model(piece, restart_preload=False)
            self._refresh_piece_list_item(piece_index)
            if piece_index == current_row:
                self._refresh_piece_header(piece_index)
                for row_index in changed_rows:
                    self._refresh_visible_review_row_status(piece_index, row_index)
            else:
                self._invalidate_piece_page_for_refresh(piece_index)

        try:
            if undone <= 0:
                self.save_status_label.setText("No manually completed SDLXLIFF sidecars selected")
            else:
                self.save_status_label.setText(
                    f"Undid completed mark for {undone} SDLXLIFF sidecar{'s' if undone != 1 else ''}"
                )
        except Exception:
            pass

    # Compatibility for callers retaining references to the former action
    # names. The visible command now represents progress completion.
    def _mark_review_sidecars_green(self, piece_rows):
        return self._mark_review_sidecars_completed(piece_rows)

    def _undo_review_sidecars_green(self, piece_rows):
        return self._undo_review_sidecars_completed(piece_rows)

    def _piece_tooltip_work(self, piece_index):
        if piece_index < 0 or piece_index >= len(self.pieces):
            return []
        piece = self.pieces[piece_index]
        work = []
        for row_idx, row_data in enumerate(piece.get("rows") or []):
            source_text = str(row_data.get("source", "") or "").strip()
            if not source_text:
                continue
            work.append((
                row_idx,
                self._tooltip_translation_key(piece, row_data),
                source_text,
                row_data.get("source_tag"),
            ))
        return work

    def _mark_tooltip_translation_pending(self, piece_index, work, refresh=True):
        try:
            if piece_index < 0 or piece_index >= len(self.pieces):
                return
            rows = self.pieces[piece_index].get("rows") or []
            pending_rows = []
            for row_idx, _key, _source_text, _tag_name in work:
                if 0 <= row_idx < len(rows):
                    rows[row_idx]["tooltip_translation_pending"] = True
                    rows[row_idx]["tooltip_translation_status"] = self._machine_translation_pending_text()
                    rows[row_idx].pop("tooltip_translation_error", None)
                    rows[row_idx].pop("tooltip_translation_error_detail", None)
                    rows[row_idx]["_source_preview_dirty"] = True
                    pending_rows.append(row_idx)
            if pending_rows:
                self._invalidate_piece_render_model(self.pieces[piece_index], restart_preload=False)
                self._refresh_open_notepad_machine_translation_context()
            if refresh and pending_rows:
                if self._review_context_menu_is_open():
                    self._queue_refresh_current_visible_dirty_source_previews()
                else:
                    self._update_review_row_source_previews(piece_index, pending_rows, visible_only=True)
        except Exception:
            pass

    def _apply_tooltip_translation_status(self, piece_index, keys, message):
        try:
            if piece_index < 0 or piece_index >= len(self.pieces):
                return
            message = str(message or "").strip()
            if not message:
                return
            key_set = set(keys or [])
            piece = self.pieces[piece_index]
            rows = piece.get("rows") or []
            changed_rows = []
            for row_index, row_data in enumerate(rows):
                if key_set and self._tooltip_translation_key(piece, row_data) not in key_set:
                    continue
                if not row_data.get("tooltip_translation_pending"):
                    continue
                row_data["tooltip_translation_status"] = message
                row_data["_source_preview_dirty"] = True
                changed_rows.append(row_index)
            if not changed_rows:
                return
            self._invalidate_piece_render_model(piece, restart_preload=False)
            self._refresh_open_notepad_machine_translation_context()
            if piece_index == self._displayed_piece_row():
                if self._review_context_menu_is_open():
                    self._queue_refresh_current_visible_dirty_source_previews()
                else:
                    self._update_review_row_source_previews(piece_index, changed_rows, visible_only=True)
        except Exception:
            pass

    def _schedule_target_edit(self, piece_index, row_index, text):
        if piece_index < 0 or row_index < 0:
            return
        self._pending_target_edits[(piece_index, row_index)] = text
        self.save_status_label.setText("Unsaved")
        self._edit_save_timer.start(500)

    def _schedule_notepad_document_edit(
        self,
        piece_index,
        text,
        user_added_target_indexes=None,
        user_added_break_positions=None,
    ):
        if piece_index < 0 or piece_index >= len(self.pieces):
            return
        self._pending_notepad_edits[piece_index] = (
            str(text or ""),
            tuple(sorted({
                int(value)
                for value in (user_added_target_indexes or [])
                if int(value) >= 0
            })),
            tuple(sorted(
                (
                    int(raw_index),
                    tuple(sorted({
                        int(position)
                        for position in raw_positions
                        if int(position) >= 0
                    })),
                )
                for raw_index, raw_positions in dict(
                    user_added_break_positions or {}
                ).items()
                if int(raw_index) >= 0
            )),
        )
        self.save_status_label.setText("Unsaved HTML")
        self._edit_save_timer.start(500)

    def _flush_target_edits(self):
        pending = dict(self._pending_target_edits)
        self._pending_target_edits.clear()
        pending_notepad = dict(self._pending_notepad_edits)
        self._pending_notepad_edits.clear()
        if not pending and not pending_notepad:
            return
        saved = 0
        try:
            for (piece_index, row_index), text in pending.items():
                if self._apply_target_edit(piece_index, row_index, text):
                    saved += 1
            for piece_index, notepad_edit in pending_notepad.items():
                if isinstance(notepad_edit, tuple) and len(notepad_edit) == 3:
                    (
                        html_text,
                        user_added_target_indexes,
                        stored_break_positions,
                    ) = notepad_edit
                    user_added_break_positions = dict(stored_break_positions)
                elif isinstance(notepad_edit, tuple):
                    html_text, user_added_target_indexes = notepad_edit
                    user_added_break_positions = None
                else:
                    html_text, user_added_target_indexes = notepad_edit, None
                    user_added_break_positions = None
                if self._apply_notepad_document_edit(
                    piece_index,
                    html_text,
                    user_added_target_indexes=user_added_target_indexes,
                    user_added_break_positions=user_added_break_positions,
                ):
                    saved += 1
            self.save_status_label.setText("Saved" if saved else "")
        except Exception as exc:
            self.save_status_label.setText(f"Save failed: {exc}")

    def _apply_notepad_document_edit(
        self,
        piece_index,
        html_text,
        *,
        user_added_target_indexes=None,
        user_added_break_positions=None,
    ):
        """Save one browser-edited HTML document and rebuild its analysis."""
        if piece_index < 0 or piece_index >= len(self.pieces):
            return False
        piece = self.pieces[piece_index]
        html_text = str(html_text or "")
        normalized_user_indexes = sorted({
            int(value)
            for value in (user_added_target_indexes or [])
            if int(value) >= 0
        })
        normalized_break_positions = {
            int(raw_index): sorted({
                int(position)
                for position in raw_positions
                if int(position) >= 0
            })
            for raw_index, raw_positions in dict(
                user_added_break_positions or {}
            ).items()
            if int(raw_index) >= 0
        }
        if (
            html_text == self._unescape_html_document(piece.get("target_html") or "")
            and normalized_user_indexes
                == sorted(piece.get("user_added_target_indexes") or [])
            and normalized_break_positions
                == dict(piece.get("user_added_break_positions") or {})
        ):
            return True

        # Track the current browser DOM order independently from the SDLXLIFF
        # text-unit rows.  This also includes empty/structural elements.
        tag_order = []
        try:
            from bs4 import BeautifulSoup
            parsed = BeautifulSoup(self._unescape_html_document(html_text), "html.parser")
            tag_order = [str(tag.name or "").casefold() for tag in parsed.find_all(True)]
        except Exception:
            tag_order = []

        saved_html = self._write_piece_target_html(
            piece,
            html_text,
            user_added_target_indexes=user_added_target_indexes,
            user_added_break_positions=user_added_break_positions,
        )
        piece["target_html"] = saved_html
        metadata = {
            "output_name": piece.get("output_name"),
            "original_name": piece.get("original_name"),
            "progress_key": piece.get("progress_key"),
            "opf_position": piece.get("opf_position"),
            "chapter_num": piece.get("chapter_num"),
            "sort_key": piece.get("sort_key"),
            "display_position": (
                int(piece.get("opf_position")) + 1
                if piece.get("opf_position") is not None
                else piece_index + 1
            ),
            "label": piece.get("review_label"),
            "progress_entry": {
                "manual_editing_pending": piece.get("manual_editing_pending", False),
            },
        }
        rebuilt = self._build_piece(piece.get("path"), piece_index, metadata)
        rebuilt["_notepad_tag_order"] = tag_order
        old_rows = piece.get("rows") or []
        old_rows_by_source_index = {
            row.get("source_index"): row
            for row in old_rows
            if row.get("source_index") is not None
        }
        old_added_rows_by_target_index = {
            row.get("target_index"): row
            for row in old_rows
            if row.get("source_index") is None and row.get("target_index") is not None
        }
        for row_data in rebuilt.get("rows") or []:
            old_row = None
            source_index = row_data.get("source_index")
            if source_index is not None:
                old_row = old_rows_by_source_index.get(source_index)
            elif row_data.get("target_index") is not None:
                old_row = old_added_rows_by_target_index.get(row_data.get("target_index"))
            if old_row is not None:
                row_data["target_original"] = old_row.get(
                    "target_original", row_data.get("target_original", "")
                )
                # Notepad saves rebuild the complete piece from its SDLXLIFF
                # and output HTML. Keep the live machine-preview state on the
                # replacement row just as compact-mode target edits do. The
                # JSON sidecar remains the durable cache, but an editor save
                # must not make the visible preview depend on another disk
                # read (or discard an in-flight/error state).
                for field in (
                    "tooltip_translation",
                    "tooltip_translation_pending",
                    "tooltip_translation_status",
                    "tooltip_translation_error",
                    "tooltip_translation_error_detail",
                    "_source_preview_dirty",
                ):
                    if field in old_row:
                        row_data[field] = old_row[field]
        self.pieces[piece_index] = rebuilt
        self._refresh_piece_list_item(piece_index)
        self._refresh_piece_header(piece_index)

        # Image rename normalization can legitimately alter the document
        # during save. Reflect it in the active rendered surface.
        self._refresh_notepad_page_after_save(piece_index, rebuilt, saved_html, html_text)
        try:
            self._last_review_signature = self._current_review_signature()
        except Exception:
            pass
        return True

    def _output_name_for_piece(self, piece):
        output_name = piece.get("output_name") or self._sidecar_output_name(piece.get("path") or piece.get("name") or "")
        if piece.get("manual_editing"):
            output_name = self._manual_output_name_for_piece(piece, output_name)
        return output_name or piece.get("name") or "SDLXLIFF output"

    def _output_path_for_piece(self, piece):
        output_name = self._output_name_for_piece(piece)
        if not output_name:
            return None
        return os.path.join(self.output_dir or "", output_name)

    def _html_with_output_image_renames(self, target_html):
        """Apply the workspace's canonical image names to generated HTML."""
        rename_map_path = os.path.join(
            self.output_dir or "", "image_rename_map.json"
        )
        try:
            with open(rename_map_path, "r", encoding="utf-8") as handle:
                rename_map = json.load(handle)
        except (OSError, ValueError, TypeError):
            return target_html
        if not isinstance(rename_map, dict) or not rename_map:
            return target_html

        try:
            from bs4 import BeautifulSoup
            from Chapter_Extractor import _update_image_refs_in_soup

            soup = BeautifulSoup(target_html or "", "html.parser")
            if _update_image_refs_in_soup(soup, rename_map):
                return str(soup)
        except Exception:
            pass
        return target_html

    def _target_html_with_edit(self, piece, row_data, text):
        try:
            from bs4 import BeautifulSoup
        except Exception:
            return piece.get("target_html", "")
        # NOTE: must use the guarded unescape — blanket-unescaping a real
        # HTML document turns escaped text (&lt;Prologue&gt;) into phantom
        # tags, and since this function WRITES the re-serialized soup back
        # to the sidecar, that would permanently bake the corruption in.
        target_html = self._unescape_html_document(piece.get("target_html"))
        soup = BeautifulSoup(target_html, "html.parser")

        def _is_editable_node(tag):
            name = str(getattr(tag, "name", "") or "").casefold()
            if name in self.TEXT_TAGS:
                return True
            if name != "div":
                return False
            classes = tag.get("class") or []
            if isinstance(classes, str):
                classes = classes.split()
            return "u" in {str(value).casefold() for value in classes} and not tag.find(self.TEXT_TAGS)

        tag_nodes = list(soup.find_all(_is_editable_node))
        target_index = row_data.get("target_index")
        node = None
        if isinstance(target_index, int) and 0 <= target_index < len(tag_nodes):
            node = tag_nodes[target_index]
        elif row_data.get("target_tag"):
            editable_nodes = [tag for tag in tag_nodes if tag.get_text(" ", strip=True)]
            fallback_index = min(len(editable_nodes) - 1, max(0, int(row_data.get("row_index", 0)))) if editable_nodes else -1
            if fallback_index >= 0:
                node = editable_nodes[fallback_index]

        if node is None:
            tag_name = row_data.get("source_tag") or row_data.get("target_tag") or "p"
            node = soup.new_tag(tag_name)
            if soup.body:
                soup.body.append(node)
            else:
                soup.append(node)
            row_data["target_tag"] = tag_name
            # Mirror the source label (with its ordinal) so the row caption
            # renders as e.g. "p(3)" instead of "p(3) -> p" after the edit.
            row_data["target_tag_label"] = row_data.get("source_tag_label") or tag_name
            row_data["target_index"] = len(list(soup.find_all(_is_editable_node))) - 1

        # <hr> is exposed to the reviewer as a paragraph-like ***** unit. If
        # the user edits that unit, replace the void element with a real <p>
        # instead of producing invalid <hr>text</hr> markup.
        if str(getattr(node, "name", "") or "").lower() == "hr":
            replacement = soup.new_tag("p")
            node.replace_with(replacement)
            node = replacement
            row_data["target_tag"] = "p"

        node.clear()
        node.append(str(text or ""))
        return str(soup)

    def _write_piece_target_html(
        self,
        piece,
        target_html,
        *,
        user_added_target_indexes=None,
        user_added_break_positions=None,
    ):
        sidecar_path = piece.get("path")
        if not sidecar_path:
            return target_html
        previous_output_name = (
            piece.get("output_name")
            or self._sidecar_output_name(sidecar_path)
        )
        target_html = self._html_with_output_image_renames(target_html)
        tree = ET.parse(sidecar_path)
        root = tree.getroot()
        target_element = None
        for element in root.iter():
            if self._local_name(element.tag) == "target":
                target_element = element
                break
        if target_element is None:
            raise ValueError("SDLXLIFF target element not found")
        for child in list(target_element):
            target_element.remove(child)
        target_element.text = target_html
        if user_added_target_indexes is not None:
            stored_indexes = sorted({
                int(value)
                for value in user_added_target_indexes
                if int(value) >= 0
            })
            for element in root.iter():
                if self._local_name(element.tag) != "file":
                    continue
                if stored_indexes:
                    element.set(
                        USER_ADDED_TARGET_INDEXES_ATTRIBUTE,
                        json.dumps(stored_indexes, separators=(",", ":")),
                    )
                else:
                    element.attrib.pop(USER_ADDED_TARGET_INDEXES_ATTRIBUTE, None)
                break
        if user_added_break_positions is not None:
            stored_positions = {}
            for raw_index, raw_positions in dict(
                user_added_break_positions or {}
            ).items():
                index = int(raw_index)
                positions = sorted({
                    int(position)
                    for position in raw_positions
                    if int(position) >= 0
                })
                if index >= 0 and positions:
                    stored_positions[str(index)] = positions
            for element in root.iter():
                if self._local_name(element.tag) != "file":
                    continue
                if stored_positions:
                    element.set(
                        USER_ADDED_BREAK_POSITIONS_ATTRIBUTE,
                        json.dumps(stored_positions, separators=(",", ":")),
                    )
                else:
                    element.attrib.pop(USER_ADDED_BREAK_POSITIONS_ATTRIBUTE, None)
                break
        was_manual_untranslated = _is_manual_untranslated_sdlxliff(root)
        is_manual_editing = (
            was_manual_untranslated
            or piece.get("manual_editing")
            or _is_manual_editing_sdlxliff(root)
        )
        try:
            ET.register_namespace("", "urn:oasis:names:tc:xliff:document:1.2")
            ET.register_namespace("sdl", "http://sdl.com/FileTypes/SdlXliff/1.0")
            ET.register_namespace("glossarion", "urn:glossarion:sdlxliff")
        except Exception:
            pass
        tree.write(sidecar_path, encoding="utf-8", xml_declaration=True)

        if is_manual_editing:
            piece["output_name"] = self._manual_output_name_for_piece(
                piece,
                previous_output_name,
            )
            self._mark_piece_progress_pending(piece, previous_output_name)
        output_path = self._output_path_for_piece(piece)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(target_html)
        if was_manual_untranslated:
            _clear_manual_untranslated_sdlxliff(root)
            tree.write(sidecar_path, encoding="utf-8", xml_declaration=True)
            piece["manual_untranslated"] = False
            piece["manual_editing"] = True

            desired_sidecar_path = os.path.join(
                os.path.dirname(sidecar_path),
                f"{piece.get('output_name')}.sdlxliff",
            )
            if (
                os.path.normcase(os.path.abspath(desired_sidecar_path))
                != os.path.normcase(os.path.abspath(sidecar_path))
                and not os.path.exists(desired_sidecar_path)
            ):
                os.rename(sidecar_path, desired_sidecar_path)
                piece["path"] = desired_sidecar_path
                piece["name"] = os.path.basename(desired_sidecar_path)
                if getattr(self, "current_path", "") and (
                    os.path.normcase(os.path.abspath(self.current_path))
                    == os.path.normcase(os.path.abspath(sidecar_path))
                ):
                    self.current_path = desired_sidecar_path
        try:
            self._last_review_signature = self._current_review_signature()
        except Exception:
            pass
        try:
            # This method writes both the sidecar and its output HTML. Record
            # the resulting auto-generation signature so the background
            # watcher does not mistake our own save for an external output
            # change and immediately regenerate what was just edited.
            self._last_autogen_signature = self._current_review_autogen_signature()
        except Exception:
            pass
        return target_html

    def _apply_target_edit(self, piece_index, row_index, text):
        if piece_index < 0 or piece_index >= len(self.pieces):
            return False
        piece = self.pieces[piece_index]
        rows = piece.get("rows") or []
        if row_index < 0 or row_index >= len(rows):
            return False
        row_data = rows[row_index]
        row_data["row_index"] = row_index
        before = [
            (str(existing.get("status") or ""), str(existing.get("reason") or ""))
            for existing in rows
        ]
        target_html = self._target_html_with_edit(piece, row_data, text)
        target_html = self._write_piece_target_html(piece, target_html)
        piece["target_html"] = target_html
        row_data["target"] = str(text or "")
        manual_cleared = self._clear_piece_manual_green_override(piece, persist=True)
        if manual_cleared:
            self._recompute_piece_row_statuses(piece)
        else:
            status, reason = self._row_status(
                row_data.get("source", ""),
                row_data.get("target", ""),
                source_missing=bool(row_data.get(
                    "source_missing", not row_data.get("source_tag")
                )),
                target_missing=bool(row_data.get(
                    "target_missing", not row_data.get("target_tag")
                )),
            )
            if row_data.get("source_tag") and row_data.get("target_tag") and row_data.get("source_tag") != row_data.get("target_tag"):
                status, reason = self._tag_mismatch_status(row_data.get("source_tag"), row_data.get("target_tag"))
            row_data["status"] = status
            row_data["reason"] = reason
        self._invalidate_piece_render_model(piece, restart_preload=False)
        self._refresh_piece_summary(piece)
        self._refresh_piece_list_item(piece_index)
        self._refresh_piece_header(piece_index)
        changed_rows = [
            idx for idx, existing in enumerate(rows)
            if idx >= len(before)
            or before[idx] != (str(existing.get("status") or ""), str(existing.get("reason") or ""))
        ]
        if row_index not in changed_rows:
            changed_rows.append(row_index)
        for changed_row in changed_rows:
            self._refresh_visible_review_row_status(piece_index, changed_row)
        return True

    @staticmethod
    def _review_layout_trailing_stretch_index(layout):
        """Index of the trailing stretch spacer in a review rows layout, or -1."""
        try:
            count = layout.count()
            if count:
                item = layout.itemAt(count - 1)
                if item is not None and item.spacerItem() is not None:
                    return count - 1
        except Exception:
            pass
        return -1

    def _review_row_text_heights(self, row_height, source_lines=1, target_lines=1, tooltip_lines=0):
        source_lines = max(1, int(source_lines or 1))
        target_lines = max(1, int(target_lines or 1))
        tooltip_lines = max(0, int(tooltip_lines or 0))
        source_height = max(34, (source_lines + tooltip_lines) * 22 + (28 if tooltip_lines else 10))
        target_height = max(self.REVIEW_TARGET_EDIT_MIN_HEIGHT, target_lines * 22 + 28)
        available = max(self.REVIEW_TARGET_EDIT_MIN_HEIGHT, int(row_height or self.REVIEW_ROW_MIN_HEIGHT) - 10)
        return min(source_height, available), min(target_height, available)

    def _inject_current_machine_translation_to_target(self, piece_index, row_index, editor=None):
        try:
            piece = self.pieces[piece_index]
            row_data = (piece.get("rows") or [])[row_index]
            translated = self._row_tooltip_translation(piece, row_data)
            if not str(translated or "").strip():
                self.save_status_label.setText("No machine translation preview")
                return
            self._inject_machine_translation_to_target(piece_index, row_index, translated, editor)
        except Exception as exc:
            try:
                self.save_status_label.setText(f"Inject failed: {exc}")
            except Exception:
                pass

    def _notepad_initial_document_html(self, piece, *, fill_untranslated=True):
        """Attach source metadata and optionally show source markup for blank targets."""
        source_html = self._unescape_html_document(piece.get("source_html") or "")
        target_html = self._unescape_html_document(piece.get("target_html") or "")
        try:
            from bs4 import BeautifulSoup

            source_soup = BeautifulSoup(source_html, "html.parser")
            target_soup = BeautifulSoup(target_html or source_html, "html.parser")

            def _is_review_text_node(tag):
                name = str(getattr(tag, "name", "") or "").casefold()
                if name in self.TEXT_TAGS:
                    return True
                if name != "div":
                    return False
                classes = tag.get("class") or []
                if isinstance(classes, str):
                    classes = classes.split()
                return "u" in {str(value).casefold() for value in classes} and not tag.find(self.TEXT_TAGS)

            source_nodes = list(source_soup.find_all(_is_review_text_node))
            target_nodes = list(target_soup.find_all(_is_review_text_node))
            rows_by_target_index = {
                row_data.get("target_index"): (row_index, row_data)
                for row_index, row_data in enumerate(piece.get("rows") or [])
                if isinstance(row_data.get("target_index"), int)
            }
            user_added_indexes = set(piece.get("user_added_target_indexes") or [])
            user_added_break_positions = {}
            for raw_index, raw_positions in dict(
                piece.get("user_added_break_positions") or {}
            ).items():
                try:
                    index = int(raw_index)
                    positions = {
                        int(position)
                        for position in raw_positions
                        if int(position) >= 0
                    }
                except (TypeError, ValueError):
                    continue
                if index >= 0 and positions:
                    user_added_break_positions[index] = positions

            def _remove_user_block_trailing_breaks(target_node):
                """Drop BR-only tails; Enter-at-end is represented by a sibling P."""
                removed = 0
                while True:
                    meaningful_children = [
                        child for child in target_node.contents
                        if not (isinstance(child, str) and not child.strip())
                    ]
                    if not meaningful_children:
                        break
                    tail = meaningful_children[-1]
                    if str(getattr(tail, "name", "") or "").casefold() != "br":
                        break
                    tail.decompose()
                    removed += 1
                return removed

            for node_index, target_node in enumerate(target_nodes):
                row_match = rows_by_target_index.get(node_index)
                row_data = row_match[1] if row_match is not None else None
                user_block = bool(
                    node_index in user_added_indexes
                    or (row_data and row_data.get("translator_note"))
                )
                source_index = (
                    row_data.get("source_index")
                    if row_data is not None
                    else node_index
                )
                source_node = (
                    source_nodes[source_index]
                    if isinstance(source_index, int)
                    and 0 <= source_index < len(source_nodes)
                    else None
                )
                source_text = (
                    str(row_data.get("source") or "")
                    if row_data is not None
                    else self._normalize_review_text(
                        source_node.get_text(" ", strip=True)
                    ) if source_node is not None else ""
                )
                target_text = self._normalize_review_text(
                    target_node.get_text(" ", strip=True)
                )
                if user_block and target_text:
                    # Notepad represents Enter at the end as a separate user
                    # paragraph. Therefore a filled TN ending in <br> can only
                    # be a stale empty-slot placeholder from the source DOM.
                    if _remove_user_block_trailing_breaks(target_node):
                        target_node[
                            "data-sdl-notepad-normalized-placeholder"
                        ] = "1"
                source_break_count = (
                    len(source_node.find_all("br"))
                    if source_node is not None
                    else 0
                )
                if user_block and target_text:
                    # Every remaining break belongs to the translator note and
                    # must remain deletable, even when it reused a source slot.
                    source_break_count = 0
                target_breaks = list(target_node.find_all("br"))
                explicit_user_break_ids = {
                    id(target_breaks[position])
                    for position in user_added_break_positions.get(
                        node_index, set()
                    )
                    if 0 <= position < len(target_breaks)
                }
                legacy_extra_break_count = max(
                    0, len(target_breaks) - source_break_count
                )
                if (
                    source_node is not None
                    and not source_text
                    and target_text
                    and source_break_count
                ):
                    # A <br> inside a text-empty source paragraph is only its
                    # empty-block placeholder. Once that slot contains a
                    # translator note, retaining the placeholder creates a
                    # phantom trailing line every time Notepad is reopened.
                    if explicit_user_break_ids:
                        placeholder_breaks = [
                            line_break for line_break in target_breaks
                            if id(line_break) not in explicit_user_break_ids
                        ][-source_break_count:]
                    else:
                        placeholder_end = min(
                            len(target_breaks),
                            legacy_extra_break_count + source_break_count,
                        )
                        placeholder_breaks = target_breaks[
                            legacy_extra_break_count:placeholder_end
                        ]
                    for placeholder_break in placeholder_breaks:
                        placeholder_break.decompose()
                    target_breaks = [
                        line_break for line_break in target_breaks
                        if line_break.parent is not None
                    ]
                # Yellow means an explicitly user-created empty line. Older
                # sidecars have no durable break metadata, so their target-only
                # breaks remain deletable but are deliberately not presented as
                # user-created merely because source/target BR counts differ.
                for break_index, line_break in enumerate(target_breaks):
                    if id(line_break) in explicit_user_break_ids:
                        line_break["data-sdl-notepad-user-tag"] = "br"
                    elif break_index < legacy_extra_break_count:
                        line_break["data-sdl-notepad-loaded-extra-break"] = "br"
                target_node["data-sdl-notepad-source"] = source_text
                if source_node is not None and not source_text:
                    # Preserve the fact that this DOM slot has no source text.
                    # A structural <br> (or other text-empty inline markup)
                    # does not make it a translated paragraph. If the user
                    # types here, Notepad promotes the slot to a TN(N).
                    target_node["data-sdl-notepad-source-empty"] = "1"
                if user_block:
                    target_node["data-sdl-notepad-user-block"] = "1"
                if row_match is not None:
                    row_index, row_data = row_match
                    target_node["data-sdl-notepad-row-index"] = str(row_index)
                    target_node["data-sdl-notepad-status"] = str(
                        row_data.get("status") or "green"
                    )
                    reason = str(row_data.get("reason") or "")
                    if reason:
                        target_node["data-sdl-notepad-status-reason"] = reason
                if (
                    target_text
                    or not fill_untranslated
                    or source_node is None
                    or not source_text
                ):
                    continue
                # Replace the blank target node with the complete source node,
                # not only get_text(), so inline/void markup remains visible.
                replacement = copy.copy(source_node)
                replacement["data-sdl-notepad-source"] = source_text
                if row_match is not None:
                    replacement["data-sdl-notepad-row-index"] = str(row_index)
                    replacement["data-sdl-notepad-status"] = str(
                        row_data.get("status") or "green"
                    )
                    if reason:
                        replacement["data-sdl-notepad-status-reason"] = reason
                target_node.replace_with(replacement)

            # Browser mode does its own rendering. Avoid prettify(), which can
            # insert meaningful whitespace into inline/preformatted content.
            rendered = str(target_soup)
            return rendered if rendered.strip() else (target_html or source_html)
        except Exception:
            return target_html or source_html

    @staticmethod
    def _notepad_document_prefix(html_text):
        """Preserve declarations/doctype omitted by documentElement.outerHTML."""
        text = str(html_text or "")
        match = re.search(r"<html(?:\s|>)", text, flags=re.IGNORECASE)
        return text[:match.start()] if match else ""

    @classmethod
    def _notepad_user_added_target_indexes(cls, document_html):
        """Return text-unit indexes for live Notepad-created sibling blocks."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(str(document_html or ""), "html.parser")

            def _is_review_text_node(tag):
                name = str(getattr(tag, "name", "") or "").casefold()
                if name in cls.TEXT_TAGS:
                    return True
                if name != "div":
                    return False
                classes = tag.get("class") or []
                if isinstance(classes, str):
                    classes = classes.split()
                return (
                    "u" in {str(value).casefold() for value in classes}
                    and not tag.find(cls.TEXT_TAGS)
                )

            indexes = []
            for index, element in enumerate(soup.find_all(_is_review_text_node)):
                user_block = element.has_attr("data-sdl-notepad-user-block")
                filled_empty_source_slot = bool(
                    element.has_attr("data-sdl-notepad-source-empty")
                    and cls._normalize_review_text(
                        element.get_text(" ", strip=True)
                    )
                )
                if user_block or filled_empty_source_slot:
                    indexes.append(index)
            return indexes
        except Exception:
            return []

    @classmethod
    def _notepad_user_added_break_positions(cls, document_html):
        """Return exact BR ordinals explicitly inserted by the Notepad editor."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(str(document_html or ""), "html.parser")

            def _is_review_text_node(tag):
                name = str(getattr(tag, "name", "") or "").casefold()
                if name in cls.TEXT_TAGS:
                    return True
                if name != "div":
                    return False
                classes = tag.get("class") or []
                if isinstance(classes, str):
                    classes = classes.split()
                return (
                    "u" in {str(value).casefold() for value in classes}
                    and not tag.find(cls.TEXT_TAGS)
                )

            stored_positions = {}
            for index, element in enumerate(soup.find_all(_is_review_text_node)):
                positions = [
                    position
                    for position, line_break in enumerate(
                        element.find_all("br")
                    )
                    if line_break.has_attr("data-sdl-notepad-user-tag")
                ]
                if positions:
                    stored_positions[index] = positions
            return stored_positions
        except Exception:
            return {}

    @staticmethod
    def _clean_notepad_browser_html(document_html):
        """Remove editor-only wrappers while retaining their edited contents."""
        document = str(document_html or "")
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(document, "html.parser")
            guard_style = soup.find("style", id="sdl-notepad-guard-style")
            if guard_style is not None:
                guard_style.decompose()
            source_tooltip = soup.find(id="sdl-notepad-source-tooltip")
            if source_tooltip is not None:
                source_tooltip.decompose()
            for wrapper in list(soup.find_all(attrs={"data-sdl-notepad-text": True})):
                wrapper.unwrap()
            for element in soup.find_all(attrs={"data-sdl-notepad-whole-selection": True}):
                original = element.attrs.pop(
                    "data-sdl-notepad-whole-selection", "__sdl_missing__"
                )
                if original == "__sdl_missing__":
                    element.attrs.pop("contenteditable", None)
                else:
                    element["contenteditable"] = original
            for element in soup.find_all(attrs={"data-sdl-notepad-original-editable": True}):
                original = element.attrs.pop("data-sdl-notepad-original-editable", "__sdl_missing__")
                if original == "__sdl_missing__":
                    element.attrs.pop("contenteditable", None)
                else:
                    element["contenteditable"] = original
            for element in soup.find_all(attrs={"data-sdl-notepad-source": True}):
                element.attrs.pop("data-sdl-notepad-source", None)
            for marker in (
                "data-sdl-notepad-row-index",
                "data-sdl-notepad-status",
                "data-sdl-notepad-status-reason",
                "data-sdl-notepad-active-container",
                "data-sdl-notepad-multiline-container",
                "data-sdl-notepad-jump-highlight",
                "data-sdl-notepad-normalized-placeholder",
            ):
                for element in soup.find_all(attrs={marker: True}):
                    element.attrs.pop(marker, None)
            for marker in (
                "data-sdl-notepad-user-tag",
                "data-sdl-notepad-loaded-extra-break",
                "data-sdl-notepad-user-tag-container",
                "data-sdl-notepad-user-block",
                "data-sdl-notepad-source-empty",
                "data-sdl-notepad-original-had-text",
                "data-sdl-notepad-user-empty-container",
            ):
                for element in soup.find_all(attrs={marker: True}):
                    element.attrs.pop(marker, None)
            media_attributes = (
                ("data-sdl-notepad-original-src", "src"),
                ("data-sdl-notepad-original-href", "href"),
                ("data-sdl-notepad-original-xlink-href", "xlink:href"),
                ("data-sdl-notepad-original-data", "data"),
                ("data-sdl-notepad-original-poster", "poster"),
            )
            for marker, attribute in media_attributes:
                for element in soup.find_all(attrs={marker: True}):
                    original = element.attrs.pop(marker, None)
                    if original is not None:
                        element[attribute] = original
            return str(soup)
        except Exception:
            return document

    # -- split out of the dialog in U7 (the dialog calls these) ---------------------------

    def _init_review_state(
        self,
        output_dir,
        current_path=None,
        parent=None,
        config=None,
        autogen_owner=None,
        autogen_file_path=None,
        autogen_progress_data=None,
        autogen_output_files=None,
        autogen_manual_entries=None,
    ):
        """The data half of ``SDLXLIFFReviewDialog.__init__`` (RG 939-958): folders,
        config, the auto-generation owner, the discovered books and the edit queues."""
        self.output_dir = output_dir
        self.current_path = os.path.abspath(current_path) if current_path else ""
        parent_config = getattr(parent, "config", None)
        self._config = config if isinstance(config, dict) else (parent_config if isinstance(parent_config, dict) else {})
        self._context_parent = parent
        self._sdlxliff_autogen_owner = autogen_owner
        self._sdlxliff_autogen_file_path = autogen_file_path
        self._sdlxliff_autogen_progress_data = autogen_progress_data
        self._sdlxliff_autogen_output_files = list(autogen_output_files or []) or None
        self._sdlxliff_autogen_manual_entries = list(autogen_manual_entries or [])
        self._last_autogen_signature = None
        self._book_entries = self._discover_review_books(parent)
        self._book_index = self._initial_review_book_index()
        if self._book_entries:
            current_book = self._book_entries[self._book_index]
            self.output_dir = current_book.get("output_dir") or self.output_dir
            self.current_path = current_book.get("current_path") or self.current_path
        self.pieces = []
        self._pending_target_edits = {}
        self._pending_notepad_edits = {}

    def _translate_tooltip_work(self, translator, work):
        """One machine-translation request for a piece's preview rows (RG 10145-10157 /
        10220-10232): ``(translations, error)``; exceptions propagate to the caller."""
        error = ""
        batch_html = self._tooltip_batch_html(work)
        result = translator.translate(batch_html)
        result_error = str(result.get("error") or "").strip() if isinstance(result, dict) else ""
        result_note = self._machine_translation_result_note(result)
        translated_html = str(result.get("translatedText") or "").strip() if isinstance(result, dict) else ""
        translations = self._extract_tooltip_batch_translations(translated_html, work)
        translations, validation_error = self._validate_tooltip_batch_translations(translations, work)
        if result_error:
            error = result_error
            translations = {}
        elif validation_error:
            error = validation_error
        error = self._append_machine_translation_note(error, result_note)
        return translations, error

    def _store_tooltip_translations(self, row, translations, error):
        """Apply one preview result to a piece's rows and persist it to the Machine
        Translation JSON (RG 10316-10341).  Returns the changed row indices."""
        piece = self.pieces[row]
        changed_rows = set()
        rows_to_persist = []
        for row_index, row_data in enumerate(piece.get("rows") or []):
            key = self._tooltip_translation_key(piece, row_data)
            was_pending = bool(row_data.pop("tooltip_translation_pending", False))
            if was_pending:
                row_data.pop("tooltip_translation_status", None)
                changed_rows.add(row_index)
            if key in translations:
                translated = str(translations[key] or "").strip()
                row_data.pop("tooltip_translation_error", None)
                row_data.pop("tooltip_translation_error_detail", None)
                if self._normalized_machine_translation_text(translated) == self._normalized_machine_translation_text(row_data.get("source", "")):
                    changed_rows.add(row_index)
                    continue
                self._set_row_tooltip_translation(piece, row_data, translated, persist=False)
                rows_to_persist.append((row_data, translated))
                changed_rows.add(row_index)
            elif error and was_pending and not translations:
                row_data["tooltip_translation_error"] = self._compact_machine_translation_error(error)
                row_data["tooltip_translation_error_detail"] = str(error or "")
                row_data["_source_preview_dirty"] = True
                changed_rows.add(row_index)
        if rows_to_persist:
            self._write_machine_translation_entries(piece, rows_to_persist)
        return changed_rows

    def _piece_header_text(self, piece_index):
        """The detail header of one piece (RG 7385-7395)."""
        piece = self.pieces[piece_index]
        warning_count = piece.get("yellow_count", 0) + piece.get("purple_count", 0)
        status_text = "MISMATCH" if piece["mismatch"] else ("WARN" if warning_count else "OK")
        flagged = piece["red_count"] + warning_count
        output_name = self._output_name_for_piece(piece)
        review_label = piece.get("review_label") or f"[{piece_index + 1:03d}] Ch.{self._format_chapter_number(piece.get('chapter_num'))} |"
        return (
            f"{review_label} {output_name}  -  source {piece['source_count']} text units "
            f"- output {piece['target_count']} - {status_text} - {flagged} flagged rows "
            f"(ratio ~= {piece['count_ratio']:.2f})"
        )

    def _apply_machine_translation_threshold(self, value):
        """Save a Flag inaccurate threshold and report it (RG 4296-4301)."""
        threshold, saved = self._set_machine_translation_inaccuracy_threshold(value)
        try:
            suffix = "" if saved else " (not saved to config.json)"
            self.save_status_label.setText(f"MT inaccuracy threshold set to {threshold:g}{suffix}")
        except Exception:
            pass
        return threshold

    def _tooltip_translation_result_message(self, translations, error):
        """The status text of a finished preview (RG 10361-10374); "" when there is none."""
        if error and not translations:
            return f"Machine translation preview failed: {self._compact_machine_translation_error(error)}"
        if translations:
            provider_label = self._machine_translation_provider_label()
            message = f"Generated {len(translations)} {provider_label} machine translation preview(s)"
            if error:
                message = f"{message}. {self._compact_machine_translation_error(error)}"
            return message
        return ""

    # -- GUI hooks: GUI-free defaults; SDLXLIFFReviewDialog overrides each with its widget code --

    def _emit_review_generation_progress(self, payload):
        callback = getattr(self, "_review_generation_progress_callback", None)
        if not callable(callback):
            return
        try:
            callback(payload if isinstance(payload, dict) else {"message": str(payload)})
        except Exception:
            pass

    def _displayed_piece_row(self):
        try:
            return int(getattr(self, "_displayed_review_row", -1))
        except (TypeError, ValueError):
            return -1

    def _queue_review_data_preload(self, delay_ms=220):
        return None

    def _refresh_piece_list_item(self, piece_index):
        return None

    def _refresh_piece_header(self, piece_index):
        return None

    def _refresh_visible_review_row_status(self, piece_index, row_index):
        return None

    def _invalidate_piece_page_for_refresh(self, row):
        return None

    def _refresh_open_notepad_machine_translation_context(self):
        return None

    def _queue_refresh_current_visible_dirty_source_previews(self):
        return None

    def _update_review_row_source_previews(self, piece_index, row_indices, visible_only=True):
        return None

    def _update_machine_translation_button_tooltip(self):
        return None

    def _start_flag_accuracy_button_animation(self):
        return None

    def _queue_stop_flag_accuracy_button_animation(self, delay_ms=650):
        return None

    def _prepare_streaming_piece_list(self, work_items):
        return False

    def _stream_piece_list_item(self, original_index, piece):
        return None

    def _finish_streaming_piece_list(self):
        return None

    def _pump_review_loading_events(self, max_ms=8):
        return None

    def _set_loading_progress(self, value=0, total=0, text=""):
        return None

    def _refresh_notepad_page_after_save(self, piece_index, rebuilt, saved_html, html_text):
        """Reload the open Notepad page after a save (desktop: the WebEngine page)."""
        return None

    def _insert_into_review_editor(self, editor, text):
        """Put ``text`` into an open row editor (desktop: QPlainTextEdit, keeping undo);
        False when there is no editor, so the edit is saved directly."""
        return False

    def _missing_machine_translation_credentials(self, provider):
        """The desktop's message for a provider whose credentials are not configured."""
        provider = self._normalize_machine_translation_provider(provider)
        if provider == "deepl":
            if not self._machine_translation_config_value(self.MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY):
                return "DeepL requires an API key"
        elif provider == "bing":
            if not self._machine_translation_config_value(self.MACHINE_TRANSLATION_BING_API_KEY_CONFIG_KEY):
                return "Bing requires a Microsoft Translator API key"
        elif provider == "yandex":
            if not self._machine_translation_config_value(self.MACHINE_TRANSLATION_YANDEX_API_KEY_CONFIG_KEY):
                return "Yandex requires an API key"
            if not self._machine_translation_config_value(self.MACHINE_TRANSLATION_YANDEX_FOLDER_ID_CONFIG_KEY):
                return "Yandex requires a folder ID"
        return ""

    def _prompt_machine_translation_credentials(self, provider, force=False):
        """GUI-free: credentials are set beforehand (``set_machine_translation_credentials``);
        a missing one is reported in the status like the desktop prompt's cancel."""
        missing = self._missing_machine_translation_credentials(provider)
        if missing:
            try:
                self.save_status_label.setText(missing)
            except Exception:
                pass
            return False
        return True


class SdlxliffAutogenOwner(SdlxliffAutogenMixin):
    """A widget-free sidecar auto-generation owner (mobile; desktop: RetranslationMixin)."""

    def __init__(self, config=None, selected_files=None):
        self.config = config if isinstance(config, dict) else {}
        self.selected_files = list(selected_files or [])


# ---------------------------------------------------------------------------
# GUI-free reviewer session (mobile compact reviewer)
# ---------------------------------------------------------------------------


class _ReviewStatusLabel:
    """``save_status_label`` stand-in: keeps the last text and the history."""

    def __init__(self):
        self._text = ""
        self.messages = []

    def setText(self, text):
        self._text = str(text or "")
        if self._text:
            self.messages.append(self._text)

    def text(self):
        return self._text


class _ReviewEditSaveTimer:
    """``_edit_save_timer`` stand-in: ``start(ms)`` marks the pending edits due."""

    def __init__(self):
        self._due = None

    def start(self, msec=0):
        self._due = time.monotonic() + max(0, int(msec or 0)) / 1000.0

    def stop(self):
        self._due = None

    def isActive(self):
        return self._due is not None

    def remaining(self):
        if self._due is None:
            return None
        return max(0.0, self._due - time.monotonic())


class _SessionSettingsOwner:
    """A reviewer settings owner that keeps changes in ``config`` (no file write)."""

    def __init__(self, config):
        self.config = config
        self.saves = 0

    def save_config(self, show_message=False):
        self.saves += 1


class SdlxliffReviewSession(SdlxliffReviewCoreMixin):
    """The SDLXLIFF reviewer without widgets: the dialog's data, edits, Mark as
    Completed / Undo, Machine Translation preview / inject and Flag inaccurate.

    ``context_parent``: the owner whose ``config`` / ``save_config(show_message=False)``
    persist reviewer settings (desktop: the translator window).  Without one, settings
    stay in ``config`` (a ``_SessionSettingsOwner``: never the desktop's fallback write to
    ``<app dir>/config.json``).  ``autogen_owner`` generates missing sidecars (default:
    ``SdlxliffAutogenOwner(config)``).  ``progress_callback(payload)`` receives
    sidecar-generation progress.
    """

    def __init__(
        self,
        output_dir,
        current_path=None,
        config=None,
        *,
        context_parent=None,
        autogen_owner=None,
        autogen_file_path=None,
        autogen_progress_data=None,
        autogen_output_files=None,
        autogen_manual_entries=None,
        progress_callback=None,
    ):
        self.save_status_label = _ReviewStatusLabel()
        self._edit_save_timer = _ReviewEditSaveTimer()
        self._review_generation_progress_callback = progress_callback
        if context_parent is None:
            context_parent = _SessionSettingsOwner(config if isinstance(config, dict) else {})
        if autogen_owner is None:
            parent_config = getattr(context_parent, "config", None)
            owner_config = config if isinstance(config, dict) else (
                parent_config if isinstance(parent_config, dict) else {}
            )
            autogen_owner = SdlxliffAutogenOwner(owner_config)
        self._init_review_state(
            output_dir,
            current_path,
            context_parent,
            config,
            autogen_owner,
            autogen_file_path,
            autogen_progress_data,
            autogen_output_files,
            autogen_manual_entries,
        )
        self._piece_pages = {}
        self._piece_render_complete = set()
        self._review_data_preload_token = 0
        self._review_data_loaded = False
        self._review_context_menu_open = False
        self._tooltip_translation_running = False
        self._last_review_signature = None
        self._last_machine_translation_signature = None
        self._review_image_assets_output = ""
        self._notepad_mode_supported = False
        self._two_column_layout_enabled = True
        self._displayed_review_row = -1

    # -- loading -------------------------------------------------------------------------

    @property
    def status(self):
        return self.save_status_label.text()

    def refresh(self, force=False, validate=False):
        """Check the sidecars (generating missing / stale ones like the desktop
        auto-refresh) and reload the pieces when anything changed.

        Returns the scan result plus ``reloaded``.  Desktop rebuilds only changed
        pieces; the session reloads them all (no page cache).
        """
        self.flush_edits()
        result = self._build_review_refresh_scan_result(
            force=force,
            validate=validate,
            current_path=self.current_path,
            last_review_signature=self._last_review_signature,
            last_mt_signature=self._last_machine_translation_signature,
            last_autogen_signature=self._last_autogen_signature,
        )
        result["reloaded"] = False
        self._sdlxliff_autogen_output_files = None
        if result.get("error"):
            self.save_status_label.setText(f"SDLXLIFF refresh failed: {result.get('error')}")
            return result
        if result.get("stats"):
            summary = self._review_generation_summary(result["stats"])
            if summary:
                self.save_status_label.setText(summary)
        if (
            not self._review_data_loaded
            or result.get("force")
            or result.get("sidecar_changed")
            or result.get("autogen_changed")
            or result.get("sidecars_generated")
        ):
            self.reload()
            result["reloaded"] = True
        elif result.get("machine_translation_changed"):
            self._reload_machine_translation_previews(signature=result.get("machine_translation_signature"))
        self._last_review_signature = result.get("review_signature")
        self._last_machine_translation_signature = result.get("machine_translation_signature")
        self._last_autogen_signature = result.get("autogen_signature")
        return result

    def reload(self):
        """Re-read every sidecar into ``pieces`` (pending edits are saved first)."""
        self.flush_edits()
        self.pieces = self._load_pieces(stream_sidebar=False)
        self._review_data_loaded = True
        if self.pieces and not (0 <= self._displayed_review_row < len(self.pieces)):
            self._displayed_review_row = 0
            current = self._row_for_piece_path(self.current_path)
            if current is not None:
                self._displayed_review_row = current
        return self.pieces

    def changed_on_disk(self):
        """Whether the sidecars, the outputs / progress they are generated from or the MT
        previews changed since the last refresh (the 2 s poll; the signatures the desktop
        auto-refresh compares)."""
        try:
            return (
                self._current_review_signature() != self._last_review_signature
                or self._current_machine_translation_signature() != self._last_machine_translation_signature
                or self._current_review_autogen_signature() != self._last_autogen_signature
            )
        except Exception:
            return True

    @property
    def books(self):
        return list(self._book_entries or [])

    def switch_book(self, index):
        """Review another discovered book (Book selector)."""
        if not (0 <= index < len(self._book_entries or [])) or index == self._book_index:
            return False
        self.flush_edits()
        entry = self._book_entries[index]
        self._book_index = index
        self.output_dir = entry.get("output_dir") or self.output_dir
        self.current_path = entry.get("current_path") or ""
        self.pieces = []
        self._displayed_review_row = -1
        self._review_data_loaded = False
        self._last_review_signature = None
        self._last_machine_translation_signature = None
        self._last_autogen_signature = None
        return True

    def select_piece(self, piece_index):
        if 0 <= piece_index < len(self.pieces):
            self._displayed_review_row = piece_index
            self.current_path = self.pieces[piece_index].get("path") or self.current_path
        return self._displayed_review_row

    def piece_summary(self, piece_index):
        """The desktop's sidebar label and detail header of one piece, plus its counts."""
        piece = self.pieces[piece_index]
        return {
            "label": self._sidebar_label_for_piece(piece, piece_index),
            "header": self._piece_header_text(piece_index),
            "output_name": self._output_name_for_piece(piece),
            "mismatch": bool(piece.get("mismatch")),
            "red_count": int(piece.get("red_count") or 0),
            "yellow_count": int(piece.get("yellow_count") or 0),
            "purple_count": int(piece.get("purple_count") or 0),
            "source_count": piece.get("source_count"),
            "target_count": piece.get("target_count"),
            "completed": bool(piece.get("manual_green_override")),
        }

    # -- edits ---------------------------------------------------------------------------

    def edit_row(self, piece_index, row_index, text):
        """Queue an output-row edit (saved by ``flush_edits``; desktop: 500 ms later)."""
        self._schedule_target_edit(piece_index, row_index, text)

    def edit_document(self, piece_index, html_text, user_added_target_indexes=None,
                      user_added_break_positions=None):
        """Queue a whole-document (Notepad) edit of one piece."""
        self._schedule_notepad_document_edit(
            piece_index,
            html_text,
            user_added_target_indexes=user_added_target_indexes,
            user_added_break_positions=user_added_break_positions,
        )

    def flush_edits(self):
        """Save the queued edits now; returns the status text ("Saved", ...)."""
        self._edit_save_timer.stop()
        self._flush_target_edits()
        return self.status

    def save_row(self, piece_index, row_index, text):
        """Queue and save one output-row edit at once."""
        self.edit_row(piece_index, row_index, text)
        return self.flush_edits()

    def undo_row(self, piece_index, row_index):
        """Restore a row's original output text (row "Undo all edits")."""
        self.flush_edits()
        self._undo_all_target_edits(piece_index, row_index, None)
        return self.status

    def notepad_document(self, piece_index):
        """The piece's whole output HTML as the Notepad layout edits it."""
        return self._notepad_initial_document_html(self.pieces[piece_index])

    def output_path(self, piece_index):
        return self._output_path_for_piece(self.pieces[piece_index])

    # -- Mark as Completed ---------------------------------------------------------------

    def mark_completed(self, piece_indices):
        self.flush_edits()
        self._mark_review_sidecars_completed(list(piece_indices or []))
        return self.status

    def undo_completed(self, piece_indices):
        self.flush_edits()
        self._undo_review_sidecars_completed(list(piece_indices or []))
        return self.status

    # -- Machine Translation preview, inject, flag inaccurate -------------------------------

    def machine_translation_preview(self, piece_index, status_callback=None):
        """Generate the Machine Translation preview of one piece (blocking).

        Returns ``{'translated': n, 'error': text, 'message': status}``.
        """
        if self._tooltip_translation_running:
            return {"translated": 0, "error": "busy", "message": self.status}
        work = self._piece_tooltip_work(piece_index)
        if not work:
            return {"translated": 0, "error": "", "message": "Preview Ready"}
        self._tooltip_translation_running = True
        try:
            target_code = self._review_target_language_code()
            self._mark_tooltip_translation_pending(piece_index, work)
            try:
                translator = self._machine_translation_translator(target_code, status_callback=status_callback)
                translations, error = self._translate_tooltip_work(translator, work)
            except Exception as exc:
                translations, error = {}, str(exc)
            self._store_tooltip_translations(piece_index, translations, error)
            if translations:
                try:
                    self._last_machine_translation_signature = self._current_machine_translation_signature()
                except Exception:
                    pass
            message = self._tooltip_translation_result_message(translations, error)
            if message:
                self.save_status_label.setText(message)
            return {"translated": len(translations), "error": error, "message": message}
        finally:
            self._tooltip_translation_running = False

    def inject_machine_translation(self, piece_index, row_index):
        """Replace a row's output with its Machine Translation preview."""
        self.flush_edits()
        self._inject_current_machine_translation_to_target(piece_index, row_index, None)
        return self.status

    def flag_inaccurate(self, piece_index):
        """Flag rows whose output disagrees with the Machine Translation preview."""
        self.select_piece(piece_index)
        self._flag_current_piece_inaccurate_translations()
        return self.status

    @property
    def inaccuracy_threshold(self):
        return self._machine_translation_inaccuracy_threshold()

    def set_inaccuracy_threshold(self, value):
        return self._apply_machine_translation_threshold(value)

    def reset_inaccuracy_threshold(self):
        self._reset_machine_translation_threshold()
        return self._machine_translation_inaccuracy_threshold()

    @property
    def provider(self):
        return self._machine_translation_provider()

    def set_provider(self, provider):
        self._set_machine_translation_provider(provider)
        return self.status

    def set_machine_translation_credentials(self, provider, api_key=None, region=None, folder_id=None):
        """Store a provider's credentials (API keys are encrypted like the desktop's)."""
        provider = self._normalize_machine_translation_provider(provider)
        keys = {
            "deepl": (self.MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY, None),
            "bing": (self.MACHINE_TRANSLATION_BING_API_KEY_CONFIG_KEY, self.MACHINE_TRANSLATION_BING_REGION_CONFIG_KEY),
            "yandex": (self.MACHINE_TRANSLATION_YANDEX_API_KEY_CONFIG_KEY,
                       self.MACHINE_TRANSLATION_YANDEX_FOLDER_ID_CONFIG_KEY),
        }
        if provider not in keys:
            return False
        key_name, extra_name = keys[provider]
        if api_key is not None:
            self._persist_review_config_value(key_name, str(api_key or "").strip())
        extra = region if provider == "bing" else folder_id
        if extra_name and extra is not None:
            self._persist_review_config_value(extra_name, str(extra or "").strip())
        return not self._missing_machine_translation_credentials(provider)


def open_sdlxliff_review(output_dir, config=None, *, current_path=None, context_parent=None,
                         autogen_owner=None, source_path=None, progress_data=None,
                         manual_entries=None, progress_callback=None, load=True):
    """Open the reviewer for an output folder (and the other books its context lists).

    ``source_path``: the raw EPUB (sidecar generation reads its chapters);
    ``progress_data``: the Progress Manager's loaded ``translation_progress.json``
    (kept in sync by Mark as Completed); ``manual_entries``: untranslated rows for
    Manual editing sidecars.  ``load`` runs the first refresh.
    """
    session = SdlxliffReviewSession(
        output_dir,
        current_path,
        config,
        context_parent=context_parent,
        autogen_owner=autogen_owner,
        autogen_file_path=source_path,
        autogen_progress_data=progress_data,
        autogen_manual_entries=manual_entries,
        progress_callback=progress_callback,
    )
    if load:
        session.refresh()
    return session


__all__ = [
    'SdlxliffAutogenMixin',
    'SdlxliffAutogenOwner',
    'SdlxliffReviewCoreMixin',
    'SdlxliffReviewSession',
    'open_sdlxliff_review',
]
