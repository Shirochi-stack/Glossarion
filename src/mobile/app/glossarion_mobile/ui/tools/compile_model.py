"""Converter / Compile screen model (UI_SPEC §4.5; FEATURE_MAP qa-epub-pdf 41-61).

Pure Python (no Flet). The EPUB options are config keys the shared compile environment
(``run_env._build_epub_compile_env``) reads at job start; the screen renders them with the
schema tiles and writes them through ``MobileConfigStore``. This module holds:

* ``EPUB_OPTION_KEYS`` / ``IMAGE_OPTION_KEYS``: the desktop "EPUB Utilities" / output
  toggles in their desktop order (rendered as schema tiles), ``LAYOUT_CHOICES`` (the
  desktop "EPUB Layout" combo: Auto / EPUB2 / EPUB3 -> ``auto`` / ``epub2`` / ``epub3``);
* the font and CSS imports: picked files are copied into the folder the EPUB compiler
  reads (``EPUBCompiler._get_global_custom_fonts_dir``: ``<data>/custom_fonts`` on mobile)
  and the settings import folder (CSS override path);
* job specs: Compile EPUB, Compile PDF (a PDF workspace compiles with ``compile_pdf``; an
  EPUB workspace compiles its EPUB with "Create PDF after EPUB" on - the desktop's PDF path
  for EPUB books, rendered by the ``pdf_mupdf_html`` shim on mobile), Validate EPUB, Rename
  files.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from typing import Any, Iterable, Mapping, Optional, Sequence

__all__ = [
    "CSS_EXTENSIONS",
    "EPUB_OPTION_KEYS",
    "FONT_EXTENSIONS",
    "IMAGE_OPTION_KEYS",
    "LAYOUT_CHOICES",
    "compile_spec",
    "count_fonts",
    "custom_fonts_dir",
    "import_css",
    "import_fonts",
    "rename_spec",
    "validate_spec",
]

#: Desktop "EPUB Layout" combo (other_settings): label -> config value.
LAYOUT_CHOICES = (("auto", "Auto"), ("epub2", "EPUB2"), ("epub3", "EPUB3"))

#: Desktop EPUB output / utilities toggles (schema keys, desktop order). The layout, CSS file
#: and fonts rows are rendered by the screen itself; everything here is a schema tile.
EPUB_OPTION_KEYS = (
    "force_ncx_only",
    "attach_css_to_chapters",
    "epub_use_html_method",
    "retain_source_extension",
    "disable_epub_gallery",
    "disable_automatic_cover_creation",
    "skip_non_spine_special_files",
    "translate_special_files",
    "skip_unreferenced_epub_images",
)
#: EPUB image compression (Settings › Image / Output).
IMAGE_OPTION_KEYS = (
    "enable_image_compression",
    "image_compression_quality",
    "exclude_cover_compression",
    "exclude_gif_compression",
)

FONT_EXTENSIONS = (".ttf", ".otf", ".woff", ".woff2")
CSS_EXTENSIONS = (".css",)


def custom_fonts_dir() -> str:
    """The folder the EPUB compiler mirrors fonts from (``EPUBCompiler._get_global_custom_fonts_dir``).

    The compiler's own method (it ignores its instance) keeps the import and the compile on
    the same folder: ``mobile_runtime.data_dir(<app dir>)/custom_fonts``.
    """
    try:
        from epub_converter import EPUBCompiler

        return str(EPUBCompiler._get_global_custom_fonts_dir(None))
    except Exception:
        import mobile_runtime

        return os.path.join(mobile_runtime.data_dir(os.getcwd()), "custom_fonts")


def count_fonts(fonts_dir: str) -> int:
    try:
        return sum(1 for name in os.listdir(fonts_dir) if os.path.splitext(name)[1].lower() in FONT_EXTENSIONS)
    except OSError:
        return 0


def import_fonts(paths: Iterable[str], fonts_dir: str) -> int:
    """Copy font files (and every font inside ZIP archives) into ``fonts_dir``; returns the count.

    Same rules as the desktop "Load Font…" button (other_settings ``_on_load_font_clicked``):
    ``.ttf/.otf/.woff/.woff2`` are copied, ZIPs contribute their font entries (subfolders
    flattened), unreadable files are skipped.
    """
    os.makedirs(fonts_dir, exist_ok=True)
    copied = 0
    for src in paths or ():
        ext = os.path.splitext(src)[1].lower()
        if ext == ".zip":
            try:
                with zipfile.ZipFile(src, "r") as zf:
                    for entry in zf.namelist():
                        if os.path.splitext(entry)[1].lower() not in FONT_EXTENSIONS:
                            continue
                        name = os.path.basename(entry)
                        if not name:
                            continue
                        with zf.open(entry) as zin, open(os.path.join(fonts_dir, name), "wb") as zout:
                            zout.write(zin.read())
                        copied += 1
            except Exception:
                pass
        elif ext in FONT_EXTENSIONS:
            try:
                shutil.copy2(src, os.path.join(fonts_dir, os.path.basename(src)))
                copied += 1
            except Exception:
                pass
    return copied


def clear_fonts(fonts_dir: str) -> int:
    """Remove the imported fonts (font files only); returns how many were removed."""
    removed = 0
    try:
        names = os.listdir(fonts_dir)
    except OSError:
        return 0
    for name in names:
        if os.path.splitext(name)[1].lower() in FONT_EXTENSIONS:
            try:
                os.remove(os.path.join(fonts_dir, name))
                removed += 1
            except OSError:
                pass
    return removed


def import_css(path: str, import_dir: str) -> str:
    """Copy a CSS file into the app's settings import folder; returns the stored path."""
    if os.path.splitext(path)[1].lower() not in CSS_EXTENSIONS:
        raise ValueError("Pick a .css file")
    os.makedirs(import_dir, exist_ok=True)
    target = os.path.join(import_dir, os.path.basename(path))
    if os.path.abspath(path) != os.path.abspath(target):
        shutil.copy2(path, target)
    return target


def _origin(target: Any) -> dict:
    bid = getattr(target, "bid", "")
    title = getattr(target, "title", "")
    if bid:
        return {"type": "library", "bid": bid, "label": f"Library · {title}"}
    return {"type": "tools", "tool": "convert", "label": "Tools · Converter"}


def compile_spec(target: Any, fmt: str) -> Any:
    """Compile EPUB (``fmt='epub'``) or PDF (``fmt='pdf'``) for a ToolTarget's output folder."""
    from glossarion_mobile.services.jobs import JobSpec

    folder = getattr(target, "folder", "")
    if not folder:
        raise ValueError("This book has no output workspace to compile")
    title = str(getattr(target, "title", "") or os.path.basename(folder))
    kind = getattr(target, "kind", "") or "epub"
    params: dict = {"folder": folder}
    if fmt == "pdf":
        if kind == "pdf":
            return JobSpec(kind="compile_pdf", title=title, inputs=(folder,), params=params, origin=_origin(target))
        # An EPUB workspace: the EPUB compile with "Create PDF after EPUB" (enable_pdf_output).
        params["config_overrides"] = {"enable_pdf_output": True}
        params["pdf_after_epub"] = True
        return JobSpec(kind="compile_epub", title=title, inputs=(folder,), params=params, origin=_origin(target))
    if kind == "pdf":
        raise ValueError("A PDF workspace compiles to PDF")
    return JobSpec(kind="compile_epub", title=title, inputs=(folder,), params=params, origin=_origin(target))


def validate_spec(targets: Sequence[Any]) -> Any:
    from glossarion_mobile.services.jobs import JobSpec

    folders = [t.folder for t in targets if getattr(t, "folder", "")]
    if not folders:
        raise ValueError("Choose an output folder to validate")
    title = str(getattr(targets[0], "title", "") or os.path.basename(folders[0]))
    if len(folders) > 1:
        title = f"{title} +{len(folders) - 1}"
    return JobSpec(kind="validate_epub", title=title, inputs=tuple(folders),
                   params={"folders": list(folders)}, origin=_origin(targets[0]), resumable=False)


def rename_spec(target: Any, retain: Optional[bool] = None) -> Any:
    from glossarion_mobile.services.jobs import JobSpec

    folder = getattr(target, "folder", "")
    if not folder:
        raise ValueError("Choose an output folder")
    params: dict = {"folder": folder}
    if retain is not None:
        params["retain"] = bool(retain)
    return JobSpec(kind="rename_outputs", title=str(getattr(target, "title", "") or os.path.basename(folder)),
                   inputs=(folder,), params=params, origin=_origin(target), resumable=False)


def outputs_of(paths: Iterable[str], extensions: Sequence[str] = (".epub", ".pdf")) -> list:
    """Existing files among a job's outputs with the given extensions (result card rows)."""
    out: list = []
    for path in paths or ():
        if path and os.path.isfile(path) and path.lower().endswith(tuple(extensions)) and path not in out:
            out.append(path)
    return out


def result_lines(result: Mapping[str, Any]) -> list:
    return [str(line) for line in (result or {}).get("validation") or ()]
