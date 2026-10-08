"""Plan card / BatchPlanCard / TranslateSheet data (UI_SPEC §2.12.1, §2.12.5, §3.10; GUI-free, no Flet).

* ``RUN_OPTION_KEYS`` - the "Run options" tiles (schema keys of the desktop main-window run row:
  chapter range + spine order, input / output token limits, chunk size, temperature, batch
  translation, contextual + history / rolling summary, multipass, post-translation QA scan,
  Remove AI artifacts) and ``run_options_summary`` (the collapsed tile's "Batch 10 · Temp 0.3 ·
  Rolling summary").
* ``RunOverlayStore`` - "Only for this run": a ``MobileConfigStore`` view whose writes stay in
  ``values`` (they become the JobSpec ``config_overrides``) while reads fall through to the
  config; the settings tiles bind to it unchanged.
* ``plan_facts`` (blocking) - the facts line: chapters / pages, ≈ tokens, size, and the
  conversion the pipeline will make (``input_preparation.resolve_input_to_epub``'s archive rules:
  EPUB-in-ZIP passthrough, HTML chapter ZIP → EPUB, image ZIP / CBZ → EPUB, subtitle ZIP).
* ``range_preview`` (blocking) - "Choose chapters": the files a range translates, through the
  shared ``GlossaryPipelineMixin._get_spine_filenames_for_preview`` /
  ``_get_pdf_range_entries_for_preview`` (desktop 🔍 ``_preview_chapter_range_files``) and
  ``RunEnvMixin._parse_chapter_range_text``, on an attribute-only owner whose values come from
  the desktop start-up mapping (``settings_rules``), never a HeadlessOwner (no os.environ writes).
* ``batch_files`` (blocking) - Include subfolders: the files of a picked folder the desktop folder
  selection accepts (``translator_gui._is_supported_folder_input`` extensions).
* ``glossary_chip_label`` - the Plan card glossary chip with the loaded manual glossary's name.
"""

from __future__ import annotations

import copy
import os
import zipfile
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "BATCH_EXTENSIONS",
    "NO_FILES_TEXT",
    "RUN_OPTION_KEYS",
    "RunOverlayStore",
    "batch_files",
    "classify_archive",
    "glossary_chip_label",
    "parse_range",
    "plan_facts",
    "range_label",
    "range_preview",
    "run_options_summary",
]

#: The Plan card's Run options (UI_SPEC §2.12.1), in desktop main-window order.
RUN_OPTION_KEYS = (
    "chapter_range",
    "use_spine_order",
    "token_limit",
    "token_limit_disabled",
    "max_output_tokens",
    "manual_chunk_size",
    "translation_temperature",
    "disable_temperature",
    "batch_translation",
    "batch_size",
    "context_mode",  # U9: the desktop Context Mode combo (writes contextual / use_rolling_summary / mode)
    "translation_history_limit",
    "translation_history_rolling",
    "multipass_mode",
    "multipass_refinement_mode",
    "scan_phase_enabled",
    "REMOVE_AI_ARTIFACTS",
)

#: Inputs a picked folder contributes to a batch: the desktop ``browse_folder`` supported set
#: (translator_gui; "include subfolders" = ``deep_scan_var``) without ``.exe`` (RPG Maker games are
#: desktop-only; mobile picks a game folder in Tools › RPG Maker).
BATCH_EXTENSIONS = (
    ".epub", ".zip", ".cbz", ".pdf", ".html", ".htm", ".xhtml", ".txt", ".json", ".csv", ".md", ".sdlxliff", ".srt",
    ".ass", ".lrc", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".mp4",
)
#: The desktop "No Files Found" text (``browse_folder``).
NO_FILES_TEXT = ("No supported files found in:\n{folder}\n\nSupported formats: EPUB, HTML, HTM, XHTML, SDLXLIFF, SRT, "
                 "ASS, LRC, TXT, MD, PNG, JPG, JPEG, GIF, BMP, WebP, MP4")
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff")
_SUBTITLE_EXTENSIONS = (".srt", ".ass", ".lrc", ".vtt")
_MISSING = object()


# ---------------------------------------------------------------------------------------------------
# "Only for this run"
# ---------------------------------------------------------------------------------------------------


class RunOverlayStore:
    """A config view for one run: reads fall through to ``base``; writes land in ``values``.

    Only top-level keys are written (the Run options are top-level); ``unset`` removes this run's
    value so the config value shows again. Observers are the base store's (the tiles refresh on
    config changes) plus this view's own writes."""

    def __init__(self, base: Any, values: Optional[Mapping[str, Any]] = None,
                 on_change: Optional[Callable[[dict], Any]] = None) -> None:
        self.base = base
        self.values: dict = dict(copy.deepcopy(dict(values or {})))
        self.on_change = on_change
        self._observers: list = []

    @staticmethod
    def _key(key: Any) -> str:
        if isinstance(key, tuple):
            return ".".join(str(part) for part in key)
        return str(key)

    # ---- read ------------------------------------------------------------------------------------

    def get(self, key: Any, default: Any = None) -> Any:
        name = self._key(key)
        if name in self.values:
            return copy.deepcopy(self.values[name])
        return self.base.get(key, default) if self.base is not None else default

    def has(self, key: Any) -> bool:
        return self._key(key) in self.values or bool(self.base is not None and self.base.has(key))

    __contains__ = has

    def keys(self) -> list:
        keys = list(self.base.keys()) if self.base is not None else []
        return keys + [k for k in self.values if k not in keys]

    def __len__(self) -> int:
        return len(self.keys())

    def __iter__(self) -> Any:
        return iter(self.keys())

    def default_for(self, key: Any) -> Any:
        return self.base.default_for(key) if self.base is not None else None

    def effective(self, key: Any) -> Any:
        name = self._key(key)
        if name in self.values:
            return copy.deepcopy(self.values[name])
        return self.base.effective(key) if self.base is not None else None

    def is_modified(self, key: Any) -> bool:
        if self._key(key) in self.values:
            return True
        return bool(self.base is not None and self.base.is_modified(key))

    def snapshot(self) -> dict:
        data = self.base.snapshot() if self.base is not None else {}
        data.update(copy.deepcopy(self.values))
        return data

    # ---- write -----------------------------------------------------------------------------------

    def set(self, key: Any, value: Any) -> bool:
        name = self._key(key)
        if name in self.values and self.values[name] == value:
            return False
        self.values[name] = copy.deepcopy(value)
        self._changed(name, value)
        return True

    def set_many(self, values: Mapping[Any, Any]) -> list:
        return [key for key, value in values.items() if self.set(key, value)]

    def unset(self, key: Any) -> bool:
        name = self._key(key)
        if name not in self.values:
            return False
        del self.values[name]
        self._changed(name, self.base.get(key) if self.base is not None else None)
        return True

    def _changed(self, name: str, value: Any) -> None:
        for callback in list(self._observers):
            try:
                callback(name, value)
            except Exception:
                pass
        if self.on_change is not None:
            self.on_change(dict(self.values))

    # ---- observers (tiles / pages use the base store's API) --------------------------------------

    def observe_all(self, callback: Callable[[str, Any], Any]) -> Callable[[], None]:
        self._observers.append(callback)
        base_unsub = self.base.observe_all(callback) if self.base is not None and hasattr(self.base, "observe_all") else None

        def unsubscribe() -> None:
            if callback in self._observers:
                self._observers.remove(callback)
            if base_unsub is not None:
                base_unsub()

        return unsubscribe

    def observe_keys(self, keys: Iterable[str], callback: Callable[[str, Any], Any]) -> Callable[[], None]:
        wanted = set(keys)
        return self.observe_all(lambda key, value: callback(key, value) if key in wanted else None)

    def flush(self) -> bool:
        return bool(self.base.flush()) if self.base is not None and hasattr(self.base, "flush") else False

    def __getattr__(self, name: str) -> Any:  # job_running, save_error, ... of the base store
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)


def run_options_summary(get: Callable[[str], Any]) -> str:
    """The collapsed Run options line: "Ch 1-50 · Batch 10 · Temp 0.3 · Rolling summary · Multipass"."""
    parts: list = []
    chapter_range = str(get("chapter_range") or "").strip()
    if chapter_range:
        parts.append(f"Ch {chapter_range}" + (" (spine)" if get("use_spine_order") else ""))
    if get("batch_translation"):
        parts.append(f"Batch {get('batch_size') or '?'}")
    if not get("disable_temperature"):
        temperature = get("translation_temperature")
        if temperature not in (None, ""):
            parts.append(f"Temp {temperature}")
    # the Context Mode the flags represent (settings_rules.context_mode): a desktop rolling-summary config
    # keeps contextual off, so the flags are never read one by one
    from glossarion_mobile.state.setting_writes import context_mode_of

    mode = context_mode_of({key: get(key) for key in ("contextual", "use_rolling_summary", "rolling_summary_mode")
                            if get(key) is not None})
    if mode.startswith("rolling_summary"):
        parts.append("Rolling summary" + (" (append)" if mode.endswith("append") else ""))
    elif mode == "contextual_history":
        parts.append(f"History {get('translation_history_limit') or 0}")
    if get("multipass_mode"):
        parts.append("Multipass")
    if get("scan_phase_enabled"):
        parts.append("QA scan after")
    return " · ".join(parts) or "Settings › Translation defaults"


def range_label(chapter_range: Any, spine: Any = False) -> str:
    """The range chip: "All chapters" / "Ch 5–10" (spine positions: "Spine 5–10")."""
    text = str(chapter_range or "").strip()
    if not text:
        return "All chapters"
    return f"{'Spine' if spine else 'Ch'} {text.replace('-', '–')}"


def glossary_chip_label(effective: str, manual_path: Any) -> str:
    """The glossary chip: the effective mode plus the loaded manual glossary's file name."""
    name = os.path.basename(str(manual_path or "").strip())
    if not name or effective.endswith(": Off"):
        return effective
    return f"{effective} · {name}"


# ---------------------------------------------------------------------------------------------------
# facts line
# ---------------------------------------------------------------------------------------------------


def classify_archive(path: str, should_stop: Optional[Callable[[], bool]] = None) -> dict:
    """What the pipeline will do with a ZIP / CBZ (the order of ``resolve_input_to_epub``):
    ``{"kind": "epub" | "html" | "images" | "subtitles" | "unknown", "label": facts text, "count": n}``."""
    ext = os.path.splitext(path)[1].lower()
    tag = "CBZ" if ext == ".cbz" else "ZIP"
    try:
        from image_archive_epub import is_epub_zip, scan_image_archive

        if is_epub_zip(path):
            return {"kind": "epub", "label": f"{tag} holds an EPUB · used as is on start", "count": 0}
        if ext == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    subtitles = [n for n in archive.namelist() if os.path.splitext(n)[1].lower() in _SUBTITLE_EXTENSIONS]
            except Exception:
                subtitles = []
            if subtitles:
                return {"kind": "subtitles", "label": f"Subtitle ZIP · {len(subtitles)} file(s) translated into one folder",
                        "count": len(subtitles)}
        try:
            from html_archive_epub import is_html_archive

            if is_html_archive(path, should_stop=should_stop):
                return {"kind": "html", "label": f"HTML chapter {tag} → EPUB on start", "count": 0}
        except Exception:
            pass
        scan = scan_image_archive(path, should_stop=should_stop)
        if getattr(scan, "is_image_archive", False) or (getattr(scan, "image_count", 0)
                                                       and getattr(scan, "nested_archive_count", 0)):
            count = int(getattr(scan, "image_count", 0) or 0)
            return {"kind": "images", "label": f"{tag} → EPUB on start · {count} image(s)", "count": count}
    except Exception:
        pass
    return {"kind": "unknown", "label": f"{tag}: not an EPUB, HTML chapter or image archive", "count": 0}


def _epub_chapters(path: str) -> tuple:
    """(spine chapters, ≈ text characters) of an EPUB from its OPF (no full parse)."""
    try:
        import xml.etree.ElementTree as ET

        from epub_package import find_epub_opf_member

        with zipfile.ZipFile(path) as archive:
            opf = find_epub_opf_member(archive)
            if not opf:
                return 0, 0
            root = ET.fromstring(archive.read(opf))
            ns = {"opf": root.tag[1:root.tag.index("}")]} if root.tag.startswith("{") else {"opf": ""}
            manifest = {}
            for item in root.findall(".//opf:manifest/opf:item", ns):
                href = item.get("href") or ""
                if "html" in (item.get("media-type") or "").lower() or href.endswith((".html", ".xhtml", ".htm")):
                    manifest[item.get("id")] = href
            spine = [manifest[ref.get("idref")] for ref in root.findall(".//opf:spine/opf:itemref", ns)
                     if ref.get("idref") in manifest]
            base = os.path.dirname(opf)
            chars = 0
            sizes = {info.filename: info.file_size for info in archive.infolist()}
            for href in spine:
                member = os.path.normpath(os.path.join(base, href)).replace("\\", "/") if base else href
                chars += int(sizes.get(member, 0) or 0)
            return len(spine), chars
    except Exception:
        return 0, 0


def _token_text(chars: int) -> str:
    tokens = max(0, int(chars) // 4)  # markup included: a rough upper estimate (UI_SPEC "≈")
    if tokens >= 1_000_000:
        return f"≈{tokens / 1_000_000:.1f}M tokens"
    if tokens >= 1000:
        return f"≈{tokens // 1000}k tokens"
    return f"≈{tokens} tokens" if tokens else ""


def _size_text(size: Any) -> str:
    try:
        value = float(size)
    except (TypeError, ValueError):
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return ""


def plan_facts(path: str) -> dict:
    """Blocking: ``{"line": "48 chapters · ≈310k tokens · 12.4 MB", "conversion": text or "", "chapters": n,
    "kind": ext}`` for the Plan card facts line."""
    ext = os.path.splitext(str(path or ""))[1].lower()
    facts: dict = {"line": "", "conversion": "", "chapters": 0, "kind": ext.lstrip(".")}
    parts: list = []
    try:
        size = os.path.getsize(path)
    except OSError:
        size = None
    if ext == ".epub":
        chapters, chars = _epub_chapters(path)
        facts["chapters"] = chapters
        if chapters:
            parts.append(f"{chapters} chapter{'s' if chapters != 1 else ''}")
        if chars:
            parts.append(_token_text(chars))
    elif ext == ".pdf":
        try:
            import fitz

            with fitz.open(path) as document:
                pages = len(document)
            facts["chapters"] = pages
            parts.append(f"{pages} page{'s' if pages != 1 else ''}")
        except Exception:
            pass
    elif ext in (".zip", ".cbz"):
        info = classify_archive(path)
        facts["conversion"] = info["label"]
        facts["archive"] = info["kind"]
    elif ext in (".txt", ".md", ".markdown", ".html", ".htm", ".xhtml", ".srt", ".ass", ".lrc", ".csv", ".json"):
        if size:
            parts.append(_token_text(size))
    text = _size_text(size) if size is not None else ""
    if text:
        parts.append(text)
    facts["line"] = " · ".join(p for p in parts if p)
    return facts


# ---------------------------------------------------------------------------------------------------
# Choose chapters
# ---------------------------------------------------------------------------------------------------

#: owner attributes the shared preview methods read (desktop start-up values from the config)
_PREVIEW_VARS = ("special_file_keywords_var", "special_file_exact_var", "translate_all_numbered_html_var",
                 "translate_special_files_var", "pdf_use_toc_sections_var", "pdf_render_mode_var")


def _preview_owner(config: Mapping[str, Any]) -> Any:
    """An attribute-only owner for the shared preview methods (no HeadlessOwner start-up, no env)."""
    import settings_rules
    from translation_pipeline import GlossaryPipelineMixin

    class _PreviewOwner(GlossaryPipelineMixin):
        def append_log(self, message: str) -> None:  # the preview methods only print()
            pass

    owner = _PreviewOwner.__new__(_PreviewOwner)
    for name in _PREVIEW_VARS:
        try:
            setattr(owner, name, settings_rules._config_var(dict(config or {}), name))
        except Exception:
            pass
    return owner


def parse_range(text: Any) -> Optional[tuple]:
    """``RunEnvMixin._parse_chapter_range_text``: "5" -> (5, 5), "5-10" -> (5, 10), else None."""
    from run_env import RunEnvMixin

    return RunEnvMixin._parse_chapter_range_text(None, text)


def range_preview(config: Mapping[str, Any], path: str, range_text: Any, spine_mode: bool) -> dict:
    """Blocking: the desktop 🔍 preview of the files a range translates.

    ``{"ok": bool, "message": text when not ok, "title", "header", "note", "legend", "rows":
    [(label, filename, skipped)]}``; texts are the desktop dialog's."""
    parsed = parse_range(range_text)
    if not parsed:
        return {"ok": False, "message": "Enter a valid chapter range (e.g. 5 or 5-10) first.", "rows": []}
    start, end = parsed
    lower = str(path or "").lower()
    if not path or not os.path.isfile(path):
        return {"ok": False, "message": "Please select an EPUB or PDF file first.", "rows": []}
    is_pdf = lower.endswith(".pdf")
    if not (lower.endswith(".epub") or is_pdf):
        return {"ok": False, "message": "File preview is only available for EPUB and PDF files.", "rows": []}
    owner = _preview_owner(config)
    translate_special = bool(getattr(owner, "translate_special_files_var", False))
    pdf_scope = None
    pdf_total = 0
    if is_pdf:
        rows, pdf_scope, pdf_total = owner._get_pdf_range_entries_for_preview(path, start, end)
    else:
        rows = owner._get_spine_filenames_for_preview(path, start, end, spine_mode, translate_special)
    rows = [tuple(row) for row in rows or ()]
    if not rows:
        if is_pdf:
            scope_name = "bookmark sections" if pdf_scope == "bookmark" else "pages"
            message = (f"No PDF {scope_name} found in range {start}-{end}. "
                       f"The source contains {pdf_total} {scope_name}.")
        else:
            message = f"No files found in range {start}-{end}" + (" (spine order)" if spine_mode else "") + "."
        return {"ok": False, "message": message, "rows": []}
    translatable = [r for r in rows if not r[2]]
    skipped = [r for r in rows if r[2]]
    note = ""
    if is_pdf:
        title = f"PDF {'Bookmarks' if pdf_scope == 'bookmark' else 'Pages'} Range Preview ({start}-{end})"
        label = "Bookmark section" if pdf_scope == "bookmark" else "PDF page"
        unit = "section(s)" if pdf_scope == "bookmark" else "page(s)"
        header = f"{label} range {start}–{end}: {len(translatable)} {unit} will be translated"
        if pdf_scope == "bookmark":
            note = ("ℹ️ PDF ranges follow the 1-based bookmark section order. "
                    "The Spine Only toggle uses this same order for PDFs.")
        else:
            use_toc = bool(getattr(owner, "pdf_use_toc_sections_var", True))
            render_mode = str(getattr(owner, "pdf_render_mode_var", "fast_semantic") or "").strip().lower()
            if not use_toc:
                reason = "bookmark-section extraction is disabled"
            elif render_mode == "image":
                reason = "PDF image render mode is page-by-page"
            else:
                reason = "the PDF has no usable bookmarks"
            note = f"ℹ️ Showing pages because {reason}."
    else:
        title = f"Chapter Range Preview ({start}-{end})" + (" — Spine Order" if spine_mode else "")
        header = (f"{'Spine' if spine_mode else 'Chapter'} range {start}–{end}: "
                  f"{len(translatable)} file(s) will be translated")
    if skipped:
        header += f"  •  {len(skipped)} special file(s) skipped"
    legend = ("ℹ️ Special files (cover, nav, toc, etc.) are skipped because \"Translate Special Files\" is disabled."
              if skipped and not translate_special else "")
    return {"ok": True, "message": "", "title": title, "header": header, "note": note, "legend": legend,
            "rows": rows}


# ---------------------------------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------------------------------


def batch_files(folder: str, include_subfolders: bool = False, extensions: Sequence[str] = BATCH_EXTENSIONS) -> list:
    """Blocking: the supported files of a picked folder, like the desktop ``browse_folder``: the
    immediate folder (sorted names), or every subfolder with "include subfolders" (``os.walk``);
    the result is sorted."""
    wanted = {e.lower() for e in extensions}
    files: list = []
    if not folder or not os.path.isdir(folder):
        return files
    if include_subfolders:
        for root, _dirs, filenames in os.walk(folder):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in wanted:
                    files.append(os.path.join(root, filename))
    else:
        for filename in sorted(os.listdir(folder)):
            path = os.path.join(folder, filename)
            if os.path.isfile(path) and os.path.splitext(filename)[1].lower() in wanted:
                files.append(path)
    return sorted(files)
