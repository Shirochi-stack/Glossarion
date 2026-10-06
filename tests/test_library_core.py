"""U5 parity for the Library core moved out of epub_library (library_core / library_covers).

What moved (milestone U5, plan section 2): the scans and their merge, the output-root /
workspace / raw resolvers, search / sort / format / paging, the card data, Book Details
(loader, chapter filters, metadata edits, reader-open plan), the Library actions
(import, Organize / Undo, Delete, Clear raw link, output-root check), Scan for Raw and
the card covers. Module functions moved byte-for-byte; Qt dialog / thread methods moved
byte-for-byte into GUI-free mixins the Qt classes now inherit. Desktop parity is checked
four ways against ``epub_library`` at ``U5_BASE_SHA`` (``git show``, imported as a separate
module so legacy and new code run side by side):

* verbatim: every moved function / method is identical to its source, except the
  documented U5 seams (tests/parity/DISCREPANCIES.md);
* differential fuzz (``PARITY_U5_STATES``, default 500 seeded states) for the pure
  decisions: card signature, search / format / sort, scan diff, paging, card pill /
  ribbon / badge (against the legacy Qt card widget), metadata edits, Book Details strip
  and toggle text, chapter filters;
* golden file-system fixtures (``PARITY_U5_FS_STATES``, default 40): both scans + merge
  on random libraries, Organize / Undo / Delete / Clear raw link / Import / Scan for Raw,
  the reader-open plan and the metadata save run on identical trees through the legacy
  and the new dialog methods; outputs, messages, emitted path moves, registries, origins
  and the whole tree are compared;
* offscreen smoke of EpubLibraryDialog and BookDetailsDialog (legacy vs new) on a
  fixture library.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/test_library_core.py
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _p in (str(TESTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

#: Parent commit of U5 (U4 on main); the legacy oracle is ``git show`` of this commit.
U5_BASE_SHA = "20b446b06b10e60195769dd9c2c9078249300a9c"
STATES = max(1, int(os.environ.get("PARITY_U5_STATES", "500")))
FS_STATES = max(1, int(os.environ.get("PARITY_U5_FS_STATES", "40")))
SEED = int(os.environ.get("PARITY_U5_SEED", "5005"))

import library_core  # noqa: E402
import library_covers  # noqa: E402

#: The real ``_default_output_root`` (tests patch the module attribute to a temp root).
REAL_DEFAULT_OUTPUT_ROOT = library_core._default_output_root

#: Moved functions whose bodies carry a documented U5 edit (seam / Qt replacement).
ADAPTED_FUNCTIONS = {
    "library_core": {"_default_output_root"},
    "library_covers": {"_cover_cache_dir", "_download_remote_cover_image"},
    "reader_doc": {"_epub_cache_dir", "_reader_image_is_sizeable"},
}

#: (legacy Qt class, method) -> (module, mixin) for methods that existed at U5_BASE_SHA.
MOVED_METHODS = [
    ("_DualScannerThread", "run", "library_core", "DualScanMixin"),
    ("_LibraryDeleteThread", "_delete_one", "library_core", "LibraryDeleteMixin"),
    ("_LibraryDeleteThread", "run", "library_core", "LibraryDeleteMixin"),
    ("_CoverLoader", "run", "library_covers", "CoverLoaderMixin"),
    ("_RawScanWorker", "_classify", "library_core", "RawScanMixin"),
    ("_RawScanWorker", "_walk", "library_core", "RawScanMixin"),
    ("_RawScanWorker", "_book_keys", "library_core", "RawScanMixin"),
    ("_RawScanWorker", "_KIND_ALLOWED_EXTS", "library_core", "RawScanMixin"),
    ("_RawScanWorker", "_compute_matches", "library_core", "RawScanMixin"),
    ("_RawScanWorker", "run", "library_core", "RawScanMixin"),
    ("_ScanForRawDialog", "MATCH_EXACT", "library_core", "ScanForRawMixin"),
    ("_ScanForRawDialog", "MATCH_FUZZY", "library_core", "ScanForRawMixin"),
    ("_ScanForRawDialog", "_SUPPORTED_EXTS", "library_core", "ScanForRawMixin"),
    ("_ScanForRawDialog", "_derive_auto_exts", "library_core", "ScanForRawMixin"),
    ("_ScanForRawDialog", "_ext_suffixes", "library_core", "ScanForRawMixin"),
    ("EpubLibraryDialog", "_format_of_book", "library_core", "LibraryShelfMixin"),
    ("EpubLibraryDialog", "_sorted_books", "library_core", "LibraryShelfMixin"),
    ("EpubLibraryDialog", "_count_raw_movable", "library_core", "LibraryShelfMixin"),
    ("EpubLibraryDialog", "_count_trans_movable", "library_core", "LibraryShelfMixin"),
    ("EpubLibraryDialog", "_build_workspace_title_index", "library_core", "LibraryShelfMixin"),
    ("EpubLibraryDialog", "_count_library_files", "library_core", "LibraryShelfMixin"),
    ("EpubLibraryDialog", "_import_single_file", "library_core", "LibraryShelfMixin"),
    ("EpubLibraryDialog", "_filtered", "library_core", "LibraryShelfMixin"),
    ("EpubLibraryDialog", "_books_by_path", "library_core", "LibraryShelfMixin"),
    ("EpubLibraryDialog", "_book_matches_current_filters", "library_core", "LibraryShelfMixin"),
    ("EpubLibraryDialog", "_card_signature", "library_core", "LibraryShelfMixin"),
    ("EpubLibraryDialog", "_raw_is_in_library_raw", "library_core", "LibraryShelfMixin"),
    ("EpubLibraryDialog", "_library_raw_match_for_book", "library_core", "LibraryShelfMixin"),
    ("EpubLibraryDialog", "_card_has_saved_raw_link", "library_core", "LibraryShelfMixin"),
    ("EpubLibraryDialog", "_DELETE_KEYWORDS", "library_core", "LibraryShelfMixin"),
    ("EpubLibraryDialog", "_summarize_folder_contents", "library_core", "LibraryShelfMixin"),
    ("EpubLibraryDialog", "_format_delete_detail", "library_core", "LibraryShelfMixin"),
    ("_BookDetailsLoader", "run", "library_core", "BookDetailsLoaderMixin"),
    ("BookDetailsDialog", "_collect_tag_values", "library_core", "BookDetailsMixin"),
    ("BookDetailsDialog", "_metadata_author_values", "library_core", "BookDetailsMixin"),
    ("BookDetailsDialog", "_display_tag_values", "library_core", "BookDetailsMixin"),
    ("BookDetailsDialog", "_metadata_editor_values", "library_core", "BookDetailsMixin"),
    ("BookDetailsDialog", "_source_metadata_values", "library_core", "BookDetailsMixin"),
    ("BookDetailsDialog", "_visible_counts", "library_core", "BookDetailsMixin"),
    ("BookDetailsDialog", "_has_progress_context", "library_core", "BookDetailsMixin"),
    ("BookDetailsDialog", "_chapter_base_infos", "library_core", "BookDetailsMixin"),
    ("BookDetailsDialog", "_filtered_chapter_infos", "library_core", "BookDetailsMixin"),
    ("BookDetailsDialog", "_build_translated_overlay", "library_core", "BookDetailsMixin"),
    ("BookDetailsDialog", "_translated_css_dirs", "library_core", "BookDetailsMixin"),
    ("BookDetailsDialog", "_resolve_output_folder_target", "library_core", "BookDetailsMixin"),
    ("BookDetailsDialog", "_resolve_source_file_target", "library_core", "BookDetailsMixin"),
    ("BookDetailsDialog", "_resolve_translated_file_target", "library_core", "BookDetailsMixin"),
]

#: Documented text edits inside moved methods: (module, mixin, method) -> [(old, new)].
METHOD_EDITS = {
    ("library_core", "LibraryShelfMixin", "_card_has_saved_raw_link"): [
        ("EpubLibraryDialog._raw_is_in_library_raw", "LibraryShelfMixin._raw_is_in_library_raw"),
        ("EpubLibraryDialog._library_raw_match_for_book", "LibraryShelfMixin._library_raw_match_for_book"),
    ],
    ("library_core", "RawScanMixin", "_compute_matches"): [
        ("_ScanForRawDialog.MATCH_EXACT", "ScanForRawMixin.MATCH_EXACT"),
    ],
}


# =============================================================================================
# legacy oracle + fixtures
# =============================================================================================

def git_text(relpath: str, sha: str = U5_BASE_SHA) -> str:
    try:
        data = subprocess.run(["git", "show", f"{sha}:{relpath}"], cwd=str(REPO_ROOT),
                              capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            raise
        pytest.skip(f"legacy source {relpath}@{sha[:8]} unavailable: {exc}")
    return data.decode("utf-8-sig")


def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


_LEGACY = {}


def load_legacy_epub_library(tmp_root: Path):
    """``epub_library`` at U5_BASE_SHA imported as ``legacy_epub_library_u5`` (cached)."""
    if "module" in _LEGACY:
        return _LEGACY["module"]
    qapp()
    text = git_text("src/epub_library.py")
    folder = Path(tmp_root)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "legacy_epub_library_u5.py"
    path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("legacy_epub_library_u5", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _LEGACY["module"] = module
    return module


@pytest.fixture(scope="session")
def legacy(tmp_path_factory):
    return load_legacy_epub_library(tmp_path_factory.mktemp("legacy_u5"))


@pytest.fixture(scope="session")
def el():
    qapp()
    import epub_library
    return epub_library


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    """Never touch the real ~/Documents/Glossarion/Library or the real output roots."""
    library = tmp_path / "_isolated" / "Library"
    output = tmp_path / "_isolated" / "Output"
    library.mkdir(parents=True)
    output.mkdir(parents=True)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(library))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(output))
    for var in ("TRANSLATE_SPECIAL_FILES", "SPECIAL_FILE_KEYWORDS", "SPECIAL_FILE_EXACT",
                "TRANSLATE_ALL_NUMBERED_HTML", "EXTRACTION_WORKERS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(library_core, "_default_output_root", lambda: str(output))
    legacy_module = _LEGACY.get("module")
    if legacy_module is not None:
        monkeypatch.setattr(legacy_module, "_default_output_root", lambda: str(output))
    covers = tmp_path / "_isolated" / "covers"
    monkeypatch.setattr(library_covers, "_COVER_CACHE_DIR_OVERRIDE", str(covers))
    if legacy_module is not None:
        def _legacy_cover_dir(path=str(covers)):
            os.makedirs(path, exist_ok=True)
            return path
        monkeypatch.setattr(legacy_module, "_cover_cache_dir", _legacy_cover_dir)
    clear_caches()
    yield
    clear_caches()


def clear_caches():
    for module in (library_core, _LEGACY.get("module")):
        if module is None:
            continue
        for name in ("_SPINE_COUNT_CACHE", "_EPUB_SEARCH_METADATA_CACHE"):
            cache = getattr(module, name, None)
            if isinstance(cache, dict):
                cache.clear()


CONTAINER_XML = (
    '<?xml version="1.0"?><container version="1.0" '
    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
    '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
    "</rootfiles></container>"
)


def png_bytes(width: int = 4, height: int = 4) -> bytes:
    import struct
    import zlib
    raw = b"".join(b"\x00" + b"\x80\x40\x20" * width for _ in range(height))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def write_epub(path, chapters, *, title="Book", subjects=(), original_title=None, images=None,
               description="", creator="Author", language="ko", cover=True) -> str:
    """Minimal valid EPUB: ``chapters`` = [(filename, title, body_html)]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = dict(images or {})
    if cover and "cover.png" not in images:
        images["cover.png"] = png_bytes(6, 9)
    manifest = []
    spine = []
    for index, (filename, _title, _body) in enumerate(chapters):
        manifest.append(f'<item id="c{index}" href="Text/{filename}" media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="c{index}"/>')
    for index, name in enumerate(images):
        props = ' properties="cover-image"' if name == "cover.png" else ""
        manifest.append(f'<item id="i{index}" href="Images/{name}" media-type="image/png"{props}/>')
    manifest.append('<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>')
    meta = "".join(f"<dc:subject>{s}</dc:subject>" for s in subjects)
    if original_title:
        meta += f'<meta name="calibre:original_title" content="{original_title}"/>'
    if cover and "cover.png" in images:
        meta += f'<meta name="cover" content="i{list(images).index("cover.png")}"/>'
    opf = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="bid">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<dc:title>{title}</dc:title><dc:creator>{creator}</dc:creator>"
        f"<dc:language>{language}</dc:language><dc:identifier id=\"bid\">id-{abs(hash(title)) % 99991}</dc:identifier>"
        f"<dc:description>{description}</dc:description><dc:date>2024-05-01</dc:date>{meta}"
        f'</metadata><manifest>{"".join(manifest)}</manifest><spine toc="ncx">{"".join(spine)}</spine></package>'
    )
    nav = "".join(
        f'<navPoint id="n{i}" playOrder="{i + 1}"><navLabel><text>{t}</text></navLabel>'
        f'<content src="Text/{f}"/></navPoint>'
        for i, (f, t, _b) in enumerate(chapters))
    ncx = ('<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
           f"<head/><docTitle><text>{title}</text></docTitle><navMap>{nav}</navMap></ncx>")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", CONTAINER_XML)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/toc.ncx", ncx)
        for filename, chapter_title, body in chapters:
            archive.writestr(
                f"OEBPS/Text/{filename}",
                f'<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml">'
                f"<head><title>{chapter_title}</title></head><body>{body}</body></html>")
        for name, data in images.items():
            archive.writestr(f"OEBPS/Images/{name}", data)
    return str(path)


def chapter_list(rng, count: int, specials: bool = True):
    chapters = []
    if specials and rng.random() < 0.5:
        chapters.append(("title.xhtml", "Title Page", "<h1>Title</h1>"))
    for number in range(1, count + 1):
        filename = f"chapter{number:04d}.xhtml"
        chapters.append((filename, f"Chapter {number}", f"<h1>Chapter {number}</h1><p>Body {number}.</p>"))
    if specials and rng.random() < 0.2:
        chapters.append(("gallery.xhtml", "Gallery", "<p>gallery</p>"))
    return chapters


def progress_for(rng, workspace: Path, chapters, *, completed_ratio=0.5, chunked=False):
    progress = {"chapters": {}, "chapter_chunks": {}, "version": "2.1"}
    for index, (filename, title, _body) in enumerate(chapters):
        stem = os.path.splitext(filename)[0]
        roll = rng.random()
        if roll < completed_ratio:
            status = "completed"
        else:
            status = rng.choice(["pending", "in_progress", "failed", "qa_failed", "completed"])
        response = f"response_{stem}.html"
        entry = {
            "status": status,
            "original_basename": filename,
            "output_file": response,
            "actual_num": index,
            "content_hash": hashlib.md5(f"{workspace.name}{filename}".encode()).hexdigest(),
        }
        if status == "completed" and rng.random() < 0.85:
            (workspace / response).write_text(
                f"<html><head><title>Translated {title}</title></head><body><h1>Translated {title}</h1>"
                f"<p>Translated body {index}.</p></body></html>", encoding="utf-8")
        if chunked and rng.random() < 0.3:
            chunk_statuses = [rng.choice(["completed", "completed", "qa_failed", "pending", "failed"])
                              for _ in range(rng.randint(2, 4))]
            progress["chapter_chunks"][entry["content_hash"]] = {
                "schema_version": 2,
                "total": len(chunk_statuses),
                "entries": {str(i): {"status": s, "qa_issues_found": ["MISSING_TEXT"] if s == "qa_failed" else []}
                            for i, s in enumerate(chunk_statuses, 1)},
                "chunks": {str(i): f"chunk {i}" for i in range(1, len(chunk_statuses) + 1)},
                "completed": [i for i, s in enumerate(chunk_statuses, 1) if s == "completed"],
            }
        progress["chapters"][f"{index}_{stem}"] = entry
    if rng.random() < 0.3:
        progress["chapters"]["source_epub.txt"] = {"status": "completed", "original_basename": "source_epub.txt"}
    (workspace / "translation_progress.json").write_text(json.dumps(progress, indent=1), encoding="utf-8")
    return progress


BOOK_KINDS = ("not_started", "in_progress", "ready", "completed", "organized", "title_paired",
              "missing_raw", "outdated", "registered_translated", "txt", "pdf_like")


def build_library_fixture(root: Path, rng: random.Random, books: int = 8) -> dict:
    """A random Library: Raw / Translated shelves, registries, origins and workspaces."""
    library = root / "Library"
    raw_dir = library / "Raw"
    trans_dir = library / "Translated"
    output = root / "Output"
    downloads = root / "Downloads"
    for folder in (raw_dir, trans_dir, output, downloads):
        folder.mkdir(parents=True, exist_ok=True)
    raw_inputs = []
    translated_inputs = []
    origins = {"version": 3, "raw": {}, "translated": {}, "pairs": {}}
    for index in range(books):
        kind = rng.choice(BOOK_KINDS)
        name = f"Book{index:02d} {rng.choice(['Alpha', 'Beta', 'Gamma', 'Delta'])}"
        chapters = chapter_list(rng, rng.randint(1, 5))
        raw_home = raw_dir if rng.random() < 0.5 else downloads
        if kind == "txt":
            raw_path = raw_home / f"{name}.txt"
            raw_path.write_text("line one\nline two\n", encoding="utf-8")
        elif kind == "pdf_like":
            raw_path = raw_home / f"{name}.pdf"
            raw_path.write_bytes(b"%PDF-1.4\n% not really a pdf\n")
        else:
            raw_path = Path(write_epub(raw_home / f"{name}.epub", chapters, title=name,
                                       subjects=[rng.choice(["Fantasy", "Romance", "Drama"])]))
        if raw_home == downloads or rng.random() < 0.4:
            raw_inputs.append(str(raw_path))
        workspace = output / name
        if kind == "registered_translated":
            compiled = Path(write_epub(downloads / f"{name} (EN).epub", chapters, title=f"{name} EN"))
            translated_inputs.append(str(compiled))
            continue
        workspace.mkdir(parents=True, exist_ok=True)
        if rng.random() < 0.8:
            (workspace / "source_epub.txt").write_text(str(raw_path), encoding="utf-8")
        if kind == "not_started":
            (workspace / "translation_progress.json").write_text(
                json.dumps({"chapters": {}, "chapter_chunks": {}, "version": "2.1"}), encoding="utf-8")
            continue
        if kind == "outdated":
            (workspace / "translation_progress.json").write_text("{not json", encoding="utf-8")
            continue
        ratio = {"ready": 1.0, "completed": 1.0, "organized": 1.0, "title_paired": 1.0}.get(kind, 0.5)
        progress_for(rng, workspace, chapters, completed_ratio=ratio, chunked=kind == "in_progress")
        if rng.random() < 0.6:
            (workspace / "metadata.json").write_text(json.dumps({
                "title": f"{name} Translated", "original_title": name,
                "subject": rng.choice(["Fantasy", "Sci-Fi, Mystery", "#Action #Drama"]),
            }), encoding="utf-8")
        if kind in ("completed", "organized", "title_paired"):
            compiled = Path(write_epub(workspace / f"{name} Translated.epub", chapters,
                                       title=f"{name} Translated", original_title=name))
            if kind == "organized":
                target = trans_dir / compiled.name
                shutil.move(str(compiled), str(target))
                origins["translated"][target.name] = str(compiled)
                if raw_home == raw_dir:
                    origins["pairs"][target.name] = raw_path.name
            elif kind == "title_paired":
                shutil.move(str(compiled), str(trans_dir / compiled.name))
            elif rng.random() < 0.2:
                (workspace / f"{name}_translated.html").write_text("<p>dup</p>", encoding="utf-8")
        if kind == "missing_raw":
            try:
                os.remove(raw_path)
            except OSError:
                pass
    if rng.random() < 0.3:
        write_epub(trans_dir / "Loose Shelf Book.epub", chapter_list(rng, 2, specials=False), title="Loose")
    (library / "library_raw_inputs.txt").write_text("".join(p + "\n" for p in raw_inputs), encoding="utf-8")
    (library / "library_translated_inputs.txt").write_text(
        "".join(p + "\n" for p in translated_inputs), encoding="utf-8")
    (library / "library_origins.txt").write_text(json.dumps(origins, indent=2), encoding="utf-8")
    return {"library": library, "output": output, "downloads": downloads}


class Sandbox:
    """A pristine fixture copied to the same absolute path before each run (legacy / new)."""

    def __init__(self, base: Path, monkeypatch):
        self.base = Path(base)
        self.pristine = self.base / "pristine"
        self.work = self.base / "work"
        self.monkeypatch = monkeypatch

    def build(self, builder, *args, **kwargs):
        """Build in ``work`` (absolute paths inside registries / sidecars point at the
        working tree every run uses), then keep a pristine copy to reset from."""
        for folder in (self.work, self.pristine):
            if folder.exists():
                shutil.rmtree(folder)
        extra = kwargs.pop("extra", None)
        self.work.mkdir(parents=True)
        built = builder(self.work, *args, **kwargs)
        if extra is not None:
            extra(built)
        shutil.copytree(self.work, self.pristine, copy_function=shutil.copy2)
        return built

    def reset(self):
        if self.work.exists():
            shutil.rmtree(self.work)
        shutil.copytree(self.pristine, self.work, copy_function=shutil.copy2)
        self.monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(self.work / "Library"))
        self.monkeypatch.setenv("OUTPUT_DIRECTORY", str(self.work / "Output"))
        out = str(self.work / "Output")
        self.monkeypatch.setattr(library_core, "_default_output_root", lambda: out)
        legacy_module = _LEGACY.get("module")
        if legacy_module is not None:
            self.monkeypatch.setattr(legacy_module, "_default_output_root", lambda: out)
        clear_caches()
        return self.work

    def snapshot(self) -> dict:
        tree = {}
        for path in sorted(self.work.rglob("*")):
            rel = path.relative_to(self.work).as_posix()
            if path.is_dir():
                tree[rel + "/"] = None
            else:
                data = path.read_bytes()
                if len(data) > 4096 and not rel.endswith(".json"):
                    tree[rel] = hashlib.sha1(data).hexdigest()
                else:
                    text = data.decode("utf-8", "replace")
                    # chunk resets stamp time.time(); runs differ by milliseconds
                    tree[rel] = re.sub(r'"last_updated": [0-9.]+', '"last_updated": 0', text)
        return tree


def norm(value):
    """JSON-normalised (tuples -> lists, sets sorted) for equality checks."""
    if isinstance(value, dict):
        return {str(k): norm(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [norm(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(norm(v) for v in value)
    return value


# =============================================================================================
# 1. verbatim moves, re-exports, import hygiene
# =============================================================================================

def _defs(tree):
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node
    return out


@pytest.fixture(scope="module")
def legacy_tree():
    return ast.parse(git_text("src/epub_library.py"))


def _module_tree(name):
    return ast.parse((SRC / f"{name}.py").read_text(encoding="utf-8-sig"))


@pytest.mark.parametrize("module_name", ["library_core", "library_covers", "reader_doc"])
def test_moved_functions_are_verbatim(module_name, legacy_tree):
    legacy_defs = _defs(legacy_tree)
    moved = {name: node for name, node in _defs(_module_tree(module_name)).items()
             if name in legacy_defs and name != "logger" and not isinstance(node, ast.ClassDef)}
    assert len(moved) >= {"library_core": 78, "library_covers": 8, "reader_doc": 32}[module_name]
    adapted = ADAPTED_FUNCTIONS[module_name]
    for name, node in moved.items():
        if name in adapted:
            assert ast.unparse(node) != ast.unparse(legacy_defs[name]), name
            continue
        assert ast.unparse(node) == ast.unparse(legacy_defs[name]), f"{module_name}.{name} differs"


def test_adapted_functions_carry_only_the_documented_edits(legacy_tree):
    legacy_defs = _defs(legacy_tree)
    core = _defs(_module_tree("library_core"))
    covers = _defs(_module_tree("library_covers"))
    reader = _defs(_module_tree("reader_doc"))
    # _default_output_root: the LibraryEnv seam is a 3-line prefix, the rest is verbatim
    new = core["_default_output_root"]
    old = legacy_defs["_default_output_root"]
    assert [ast.dump(s) for s in new.body[2:]] == [ast.dump(s) for s in old.body[1:]]
    assert "_LIBRARY_ENV" in ast.unparse(new.body[1])
    # cache folders: the override seam only
    assert ast.unparse(covers["_cover_cache_dir"]).replace(
        "_COVER_CACHE_DIR_OVERRIDE or ", "") == ast.unparse(legacy_defs["_cover_cache_dir"])
    assert ast.unparse(reader["_epub_cache_dir"]).replace(
        "_EPUB_CACHE_DIR_OVERRIDE or ", "") == ast.unparse(legacy_defs["_epub_cache_dir"])
    # Qt image decoders replaced by header probes; everything else verbatim
    new_dl = ast.unparse(covers["_download_remote_cover_image"])
    old_dl = ast.unparse(legacy_defs["_download_remote_cover_image"])
    assert "QImage" in old_dl and "QImage" not in new_dl and "_image_bytes_decodable(data)" in new_dl
    assert new_dl.splitlines()[:20] == old_dl.splitlines()[:20]
    new_sz = ast.unparse(reader["_reader_image_is_sizeable"])
    assert "QImageReader" not in new_sz and "_probe_image_size" in new_sz
    assert new_sz.splitlines()[:5] == ast.unparse(legacy_defs["_reader_image_is_sizeable"]).splitlines()[:5]


def _class_member(tree, cls, member):
    cnode = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    for node in cnode.body:
        if isinstance(node, ast.FunctionDef) and node.name == member:
            return node
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == member for t in node.targets):
            return node
    raise KeyError((cls, member))


@pytest.mark.parametrize("cls,member,module_name,mixin", MOVED_METHODS,
                         ids=[f"{c}.{m}" for c, m, _mo, _mi in MOVED_METHODS])
def test_moved_methods_are_verbatim_in_their_mixins(cls, member, module_name, mixin, legacy_tree):
    old = _class_member(legacy_tree, cls, member)
    new_tree = _module_tree(module_name)
    new = _class_member(new_tree, mixin, member)
    old_text = ast.unparse(old)
    for before, after in METHOD_EDITS.get((module_name, mixin, member), []):
        assert before in old_text
        old_text = old_text.replace(before, after)
    if (mixin, member) == ("DualScanMixin", "run"):
        merge = _class_member(new_tree, mixin, "_merge_scan_rows")
        combined = [ast.dump(s) for s in new.body[:-1]] + [ast.dump(s) for s in merge.body[1:]]
        assert combined == [ast.dump(s) for s in old.body]
        return
    assert ast.unparse(new) == old_text


def test_desktop_classes_inherit_the_mixins_and_reexport_every_moved_name(el, legacy_tree):
    import live_stream
    import reader_doc

    modules = {"library_core": library_core, "library_covers": library_covers,
               "reader_doc": reader_doc, "live_stream": live_stream}
    for cls, member, module_name, mixin in MOVED_METHODS:
        qt_class = getattr(el, cls)
        mixin_class = getattr(modules[module_name], mixin)
        assert issubclass(qt_class, mixin_class), (cls, mixin)
        assert type(qt_class).__mro__ and qt_class.__mro__[1] is mixin_class or mixin_class in qt_class.__mro__
        assert member not in vars(qt_class), f"{cls}.{member} still defined on the Qt class"
        assert getattr(qt_class, member) is getattr(mixin_class, member) or isinstance(
            vars(mixin_class)[member], (staticmethod, classmethod)) or not callable(getattr(mixin_class, member))
    legacy_names = set(_defs(legacy_tree))
    current_defs = set(_defs(ast.parse((SRC / "epub_library.py").read_text(encoding="utf-8-sig"))))
    for module_name in ("library_core", "library_covers", "reader_doc"):
        module = modules[module_name]
        for name in _defs(_module_tree(module_name)):
            if name in legacy_names and name not in ("logger",):
                assert name not in current_defs, f"epub_library still defines {name}"
                expected = REAL_DEFAULT_OUTPUT_ROOT if name == "_default_output_root" else getattr(module, name)
                assert getattr(el, name) is expected, name


#: Desktop Library methods whose bodies were rewritten as calls into helpers extracted
#: into the mixins (DISCREPANCIES U5 "Phase-1 splits inside desktop methods"). Every other
#: method of these classes is unchanged or moved verbatim (and inherited). The reader
#: classes are pinned by tests/test_reader_doc.py.
LIBRARY_CLASSES_CHANGED = {
    "_BookCard": {"__init__"},
    "_ScanForRawDialog": {"__init__", "_populate_tree", "_apply_matches"},
    "EpubLibraryDialog": {
        "_library_page_bounds", "_update_library_pagination_controls", "_update_organize_counts",
        "_import_paths_into_library", "_ensure_output_override_matches", "_organize_into_library",
        "_undo_organize_prompt", "_on_auto_scan_done", "_clear_saved_raw_link",
        "_delete_books_prompt", "_on_delete_finished", "_confirm_delete_simple",
    },
    "_BookMetadataEditDialog": {"changed_values"},
    "BookDetailsDialog": {
        "_update_progress_strip", "_on_edit_metadata_clicked", "_update_toc_toggle_label",
        "_chapter_page_bounds", "_update_chapter_pagination_controls", "_open_reader",
    },
}
_READER_CLASSES = {"_EpubCacheLoaderThread", "_OverlayMergeThread", "_ReaderImagePreloadThread",
                   "_WorkspaceReaderLoaderThread", "_EpubSearchThread", "_EpubLoaderThread",
                   "EpubReaderDialog"}


def test_qt_library_classes_changed_only_the_documented_methods(el, legacy_tree):
    current = ast.parse((SRC / "epub_library.py").read_text(encoding="utf-8-sig"))

    def classes(tree):
        return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

    def members(cnode):
        return {node.name: ast.unparse(node) for node in cnode.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

    old_classes, new_classes = classes(legacy_tree), classes(current)
    seen = {}
    for name, old_node in old_classes.items():
        if name in _READER_CLASSES or name not in new_classes:
            continue
        old, new = members(old_node), members(new_classes[name])
        added = set(new) - set(old)
        changed = {m for m, text in new.items() if m in old and old[m] != text}
        assert not added, (name, added)
        if changed:
            seen[name] = changed
        # whatever left the Qt class is inherited from its mixin(s)
        for member in set(old) - set(new):
            assert hasattr(getattr(el, name), member), (name, member)
    assert seen == LIBRARY_CLASSES_CHANGED


_HYGIENE_PROBE = r"""
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("PySide6", "translator_gui", "dpi_setup", "epub_library", "shiboken6"):
            raise ImportError("blocked: " + name)
        return None
sys.meta_path.insert(0, Block())
sys.path.insert(0, {src!r})
import library_core, library_covers, reader_doc, live_stream, output_naming
print("OK", sorted(m for m in sys.modules if m.split(".")[0] in ("PySide6", "epub_library")))
"""


def test_shared_modules_import_without_qt_and_parse_on_python310():
    proc = subprocess.run([sys.executable, "-c", _HYGIENE_PROBE.format(src=str(SRC))],
                          capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.strip() == "OK []"
    for name in ("library_core", "library_covers", "reader_doc", "live_stream", "output_naming"):
        text = (SRC / f"{name}.py").read_text(encoding="utf-8-sig")
        tree = ast.parse(text, feature_version=(3, 10))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imported.add(node.module.split(".")[0])
        for banned in ("PySide6", "shiboken6", "translator_gui", "dpi_setup", "epub_library"):
            assert banned not in imported, (name, banned)


# =============================================================================================
# 2. differential fuzz of the pure decisions (legacy dialog methods vs shared code)
# =============================================================================================

def random_book(rng, root="C:/lib"):
    kind = rng.choice(["in_progress", "epub", "pdf", "txt", "html", "image"])
    name = rng.choice(["Alpha Book", "beta tale", "Gamma: The Story.", "델타", "Ω epsilon", ""]) + str(rng.randint(0, 9))
    book = {
        "name": name,
        "path": f"{root}/{rng.choice(['ws', 'Translated', 'x'])}/{name or 'n'}{rng.randint(0, 999)}.{kind}",
        "size": rng.choice([0, 512, 10_000, 1_048_575, 1_048_576, 5_000_000, 123_456_789]),
        "mtime": rng.choice([0, 1.5, 1700000000.25, rng.random() * 1e9]),
        "type": kind,
        "workspace_kind": rng.choice(["epub", "txt", "pdf", "image", "html", "other", "", None]),
        "translation_state": rng.choice(["", None, "not_started", "in_progress", "ready_to_compile",
                                         "completed", "outdated_progress"]),
        "is_in_progress": rng.random() < 0.7,
        "total_chapters": rng.choice([0, 1, 3, 10, 217, None]),
        "completed_chapters": rng.choice([0, 1, 2, 10, 216, None]),
        "failed_chapters": rng.choice([0, 1, None]),
        "pending_chapters": rng.choice([0, 1, 5, None]),
        "missing_raw_file": rng.random() < 0.3,
        "has_compiled_output": rng.random() < 0.4,
        "raw_source_path": rng.choice(["", f"{root}/Raw/raw{rng.randint(0, 5)}.epub", None]),
        "original_path": rng.choice(["", f"D:/orig/o{rng.randint(0, 5)}.epub"]),
        "folder_name": rng.choice(["", None, f"ws {rng.randint(0, 9)}"]),
        "compiled_conflicts": rng.choice([[], [("a.epub", "epub")], [("a", "epub"), ("b", "pdf")]]),
        "subjects": rng.choice([[], ["Fantasy"], ["Romance", "Drama"]]),
        "raw_subjects": rng.choice([[], ["판타지"]]),
        "metadata_json": rng.choice([{}, {"title": "Meta Title", "original_title": "원제"},
                                     {"subject": "#Action #Drama", "tags": ["x"]}]),
        "in_library": rng.random() < 0.3,
    }
    for key in list(book):
        if rng.random() < 0.05:
            del book[key]
    book.setdefault("name", "")
    book.setdefault("size", 0)
    book.setdefault("mtime", 0)
    book.setdefault("path", "C:/lib/fallback.epub")
    return book


class FakeSearch:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


def fake_dialog(cls, **attrs):
    obj = cls.__new__(cls)
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def test_card_and_filter_decisions_match_legacy(legacy, el):
    rng = random.Random(SEED)
    queries = ["", "alpha", "BOOK", "drama", "원제", "  ", "meta", "ω", "tale1"]
    for state in range(STATES):
        books = [random_book(rng) for _ in range(rng.randint(0, 6))]
        book = books[0] if books else random_book(rng)
        assert legacy.EpubLibraryDialog._card_signature(book) == library_core.card_signature(book)
        assert legacy.EpubLibraryDialog._format_of_book(book) == library_core.format_of_book(book)
        query = rng.choice(queries)
        fmt = rng.choice(["all", "epub", "txt", "pdf", "html", "image"])
        sort = rng.choice(["date", "name", "size"])
        attrs = dict(_search=FakeSearch(query), _format_filter=fmt, _sort_mode=sort)
        old = legacy.EpubLibraryDialog._filtered(fake_dialog(legacy.EpubLibraryDialog, **attrs), list(books))
        new = el.EpubLibraryDialog._filtered(fake_dialog(el.EpubLibraryDialog, **attrs), list(books))
        assert old == new == library_core.filter_books(books, query, fmt, sort)
        assert legacy._book_matches_library_query(book, query) == library_core.book_matches_query(book, query)
        assert legacy._card_raw_title(book) == library_core.card_raw_title(book)


def test_scan_diff_matches_legacy_auto_refresh(legacy, el):
    rng = random.Random(SEED + 1)

    class Recorder:
        def __init__(self):
            self.calls = []

    for state in range(STATES):
        old_ip = [random_book(rng) for _ in range(rng.randint(0, 4))]
        old_comp = [random_book(rng) for _ in range(rng.randint(0, 4))]
        new_ip = copy.deepcopy(old_ip)
        new_comp = copy.deepcopy(old_comp)
        for shelf in (new_ip, new_comp):
            for book in shelf:
                roll = rng.random()
                if roll < 0.2:
                    book["completed_chapters"] = (book.get("completed_chapters") or 0) + 1
                elif roll < 0.3:
                    book["name"] = book.get("name", "") + " renamed"
                elif roll < 0.35:
                    book["subjects"] = ["Changed"]
            if rng.random() < 0.15:
                shelf.append(random_book(rng))
            if shelf and rng.random() < 0.1:
                shelf.pop()
        query = rng.choice(["", "alpha", "changed", "renamed"])
        fmt = rng.choice(["all", "epub", "pdf"])
        sort = rng.choice(["date", "name", "size"])
        rec = Recorder()
        dialog = fake_dialog(
            legacy.EpubLibraryDialog, _in_progress_books=old_ip, _completed_books=old_comp,
            _search=FakeSearch(query), _format_filter=fmt, _sort_mode=sort, _card_stream_states={},
        )
        dialog._refresh_view = lambda: rec.calls.append(("refresh",))
        dialog._cards_for_tab_key = lambda key: []
        dialog._replace_mounted_library_card = lambda key, book: rec.calls.append(("replace", key, book["path"]))
        dialog._update_organize_counts = lambda: None
        legacy.EpubLibraryDialog._on_auto_scan_done(dialog, new_ip, new_comp)
        structure, changed = library_core.diff_scans(
            old_ip, old_comp, new_ip, new_comp, query=query, format_filter=fmt, sort_mode=sort)
        if structure:
            assert rec.calls == [("refresh",)]
        else:
            expected = sorted(("replace", key, path) for key in ("ip", "comp") for path in changed[key])
            assert sorted(rec.calls) == expected


def test_paging_matches_legacy_pagers(legacy):
    rng = random.Random(SEED + 2)

    class Label:
        text = None

        def setText(self, value):
            self.text = value

    class Combo:
        def __init__(self, value):
            self.value = value

        def currentData(self):
            return self.value

    for state in range(STATES):
        total = rng.choice([0, 1, 5, 19, 20, 21, 99, 100, 101, 1000])
        page = rng.choice([0, 1, 2, 5, 50, -1])
        size = rng.choice([20, 50, 100, "all", 250])
        label = Label()
        dialog = fake_dialog(
            legacy.EpubLibraryDialog, _library_pages={"ip": page},
            _library_pagers={"ip": {"page_size": Combo(size), "label": label}},
        )
        old = legacy.EpubLibraryDialog._library_page_bounds(dialog, "ip", total)
        old_page = dialog._library_pages["ip"]
        page_size = 0 if size == "all" else int(size)
        start, end, count, new_page = library_core.page_bounds(total, page, page_size)
        assert (start, end, count) == old and new_page == old_page
        dialog._library_pages["ip"] = page
        legacy.EpubLibraryDialog._update_library_pagination_controls(dialog, "ip", total)
        assert label.text == library_core.page_label(new_page, count, start, end, total, page_size)


def _card_view_from_widget(card):
    from PySide6.QtWidgets import QLabel
    labels = card.findChildren(QLabel)
    texts = sorted((lbl.text(), lbl.toolTip(), lbl.styleSheet()) for lbl in labels
                   if lbl is not card.cover_label)
    ribbon = getattr(card, "_progress_ribbon", None)
    return texts, (ribbon.text(), ribbon.styleSheet()) if ribbon is not None else None


def test_card_pill_ribbon_and_badges_match_the_legacy_widget(legacy, el):
    qapp()
    rng = random.Random(SEED + 3)
    states = max(60, STATES // 4)
    for state in range(states):
        book = random_book(rng)
        book["name"] = book.get("name") or "x"
        preset = dict(legacy._SIZE_PRESETS[rng.choice(list(legacy._SIZE_PRESETS))])
        old_card = legacy._BookCard(dict(book), preset=preset)
        new_card = el._BookCard(dict(book), preset=preset)
        try:
            assert _card_view_from_widget(old_card) == _card_view_from_widget(new_card)
            assert old_card.height() == new_card.height()
        finally:
            old_card.deleteLater()
            new_card.deleteLater()
        view = library_core.card_progress_view(book)
        if view is not None:
            assert view["ribbon_text"] in ("NOT STARTED", "IN PROGRESS", "READY TO COMPILE", "OUTDATED PROGRESS")
            assert view["pct"] == (int((view["done"] * 100) // view["total"]) if view["total"] else 0)


def test_metadata_editor_and_details_text_match_legacy(legacy, el):
    rng = random.Random(SEED + 4)
    fields = ("title", "creator", "publisher", "language", "date", "subject", "description")
    values = ["", "  ", "Title", "Title ", "A, B", "#x #y", "x\ny", None, "Ω"]
    for state in range(STATES):
        initial = {f: rng.choice(values) for f in fields if rng.random() < 0.8}
        current = {f: rng.choice(values) or "" for f in fields}
        dialog = fake_dialog(legacy._BookMetadataEditDialog, _initial_values=dict(initial))
        dialog.values = lambda current=current: dict(current)
        assert legacy._BookMetadataEditDialog.changed_values(dialog) == library_core.metadata_changed_values(
            initial, current)
        existing = {f: rng.choice(values) for f in fields if rng.random() < 0.6}
        if rng.random() < 0.3:
            existing["original_title"] = "kept"
        source = {f: rng.choice(values) for f in fields if rng.random() < 0.5}
        assert legacy._merge_manual_metadata_edits(dict(existing), dict(current), dict(source)) == \
            library_core.merge_manual_metadata_edits(dict(existing), dict(current), dict(source))


def random_chapters_info(rng):
    infos = []
    for index in range(rng.randint(0, 8)):
        info = {
            "index": index,
            "filename": rng.choice([f"ch{index}.xhtml", "title.xhtml", "gallery.xhtml", ""]),
            "raw_title": rng.choice(["Raw", "", "원문"]),
            "translated_title": rng.choice(["Trans", ""]),
            "translated_path": rng.choice(["", "x.html"]),
            "status": rng.choice(["completed", "pending", "", "qa_failed", "failed", "in_progress"]),
            "is_special": rng.random() < 0.2,
            "is_gallery": rng.random() < 0.1,
        }
        if rng.random() < 0.3:
            info["chunk_summary"] = {"total": 3, "completed": 1, "failed": rng.choice([0, 1]),
                                     "pending": rng.choice([0, 1])}
            info["chunk_status_text"] = "Chunks 1✓ 2⚠"
            info["chunks"] = [{"index": 0, "status": "qa_failed", "qa_issues_found": ["MISSING_TEXT"]}]
        infos.append(info)
    return infos


def test_book_details_decisions_match_legacy(legacy, el):
    rng = random.Random(SEED + 5)

    class Strip:
        def __init__(self):
            self.text = None
            self.visible = None

        def setText(self, value):
            self.text = value

        def hide(self):
            self.visible = False

        def show(self):
            self.visible = True

    class Toggle(Strip):
        tooltip = None

        def setToolTip(self, value):
            self.tooltip = value

    for state in range(STATES):
        infos = random_chapters_info(rng)
        book = {"is_in_progress": rng.random() < 0.7,
                "translation_state": rng.choice(["", "completed", "in_progress"]),
                "total_chapters": rng.choice([0, 3, None])}
        show_special = rng.random() < 0.5
        qa_only = rng.random() < 0.3
        search = rng.choice(["", "raw", "MISSING", "ch1", "trans"])
        common = dict(_book=book, _chapters_info=infos, _show_special_files=show_special,
                      _show_qa_failures_only=qa_only, _toc_search=FakeSearch(search),
                      _metadata_json={}, _details={})
        strip = Strip()
        toggle = Toggle()
        old = fake_dialog(legacy.BookDetailsDialog, _progress_strip=strip, _toc_toggle=toggle, **common)
        old._update_chapter_pagination_controls = lambda *a, **k: None
        legacy.BookDetailsDialog._update_progress_strip(old)
        legacy.BookDetailsDialog._update_toc_toggle_label(old)
        model = library_core.BookDetailsModel(book, {"chapters_info": infos}, {},
                                              show_special_files=show_special, search=search,
                                              qa_failures_only=qa_only)
        expected_strip = model.progress_strip_text()
        assert (strip.visible is False) == (expected_strip is None)
        if expected_strip is not None:
            assert strip.text == expected_strip
        assert (toggle.text, toggle.tooltip) == model.toggle_label()
        assert legacy.BookDetailsDialog._filtered_chapter_infos(old) == model.visible_chapters()
        assert legacy.BookDetailsDialog._visible_counts(old) == model.counts()
        for info in infos:
            for raw in (False, True):
                assert legacy._prepare_chapter_row_spec(info, raw) == library_core.prepare_chapter_row_spec(info, raw)


# =============================================================================================
# 3. golden file-system fixtures (legacy dialog methods vs the new ones on identical trees)
# =============================================================================================

class MessageBoxStub:
    """``QMessageBox`` stand-in: records texts, answers from a scripted queue."""

    Question = Warning = Information = Critical = 0
    Yes, No, Cancel, Ok = 16384, 65536, 4194304, 1024
    AcceptRole = 0
    log: list = []
    answers: list = []

    def __init__(self, *args, **kwargs):
        self._buttons = []
        self._clicked = None

    @classmethod
    def reset(cls, answers=()):
        cls.log = []
        cls.answers = list(answers)

    def setIcon(self, *_a):
        pass

    def setWindowTitle(self, title):
        self.log.append(("title", title))

    def setText(self, text):
        self.log.append(("text", text))

    def setInformativeText(self, text):
        self.log.append(("info", text))

    def setStandardButtons(self, *_a):
        pass

    def setDefaultButton(self, *_a):
        pass

    def addButton(self, label, *_a):
        button = ("button", label)
        self._buttons.append(button)
        return button

    def exec(self):
        answer = self.answers.pop(0) if self.answers else self.Yes
        if isinstance(answer, str):
            self._clicked = next((b for b in self._buttons if b[1] == answer), None)
            return 0
        return answer

    def clickedButton(self):
        return self._clicked

    @classmethod
    def information(cls, _parent, title, text, *a):
        cls.log.append(("information", title, text))
        return cls.Ok

    @classmethod
    def warning(cls, _parent, title, text, *a):
        cls.log.append(("warning", title, text))
        return cls.Ok

    @classmethod
    def question(cls, _parent, title, text, *a):
        cls.log.append(("question", title, text))
        return cls.answers.pop(0) if cls.answers else cls.Yes


class TimerStub:
    """``QTimer`` stand-in for module-level ``QTimer.singleShot`` calls (recorded, not run)."""

    shots: list = []

    @staticmethod
    def singleShot(ms, fn=None, *args):
        TimerStub.shots.append(ms)


class Emitter:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


@pytest.fixture
def qt_prompts(monkeypatch):
    legacy_module = _LEGACY.get("module")
    import epub_library
    for module in (legacy_module, epub_library):
        if module is not None:
            monkeypatch.setattr(module, "QMessageBox", MessageBoxStub)
            monkeypatch.setattr(module, "QTimer", TimerStub)
    TimerStub.shots = []
    MessageBoxStub.reset()
    yield


def shelf_dialog(module, ip, comp, config, policy="keep_both"):
    dialog = fake_dialog(module.EpubLibraryDialog, _in_progress_books=ip, _completed_books=comp,
                         _config=config, _delete_thread=None, _selected_paths_ip=set(),
                         _selected_paths_comp=set(), _delete_progress=None)
    dialog.files_reorganized = Emitter()
    dialog.loads = []
    dialog._load_books = lambda: dialog.loads.append("load")
    dialog._prompt_duplicate_policy = lambda collisions, label: (
        MessageBoxStub.log.append(("policy", sorted(collisions), label)) or policy)
    return dialog


def run_both(sandbox, legacy_module, new_module, action):
    """Run ``action(module, work)`` on a fresh copy for each side; return both results."""
    results = {}
    for label, module in (("legacy", legacy_module), ("new", new_module)):
        work = sandbox.reset()
        MessageBoxStub.reset()
        TimerStub.shots = []
        out = action(module, work)
        results[label] = {
            "out": norm(out),
            "log": norm(MessageBoxStub.log),
            "timers": list(TimerStub.shots),
            "tree": sandbox.snapshot(),
        }
    return results


def scan_order(tab):
    """One scan tab in a deterministic order: the scan's own key (mtime, newest first), then path.

    The scans append output-folder rows in thread-completion order (``as_completed``) and then
    stable-sort by mtime, so rows with exactly the same mtime come out in either order, in
    legacy and new alike (preserved desktop behaviour, EL 2848 / 2857). Every comparison that
    depends on the row order uses this order on both sides.
    """
    return sorted(tab, key=lambda row: (-float(row.get("mtime") or 0), str(row.get("path") or ""),
                                        str(row.get("output_folder") or "")))


def scan_rows(module, config):
    """The desktop scan thread (legacy or new) run synchronously, in ``scan_order``."""
    clear_caches()
    thread = module._DualScannerThread(config)
    got = []
    thread.scan_finished.connect(lambda ip, comp: got.append((ip, comp)))
    thread.run()
    ip, comp = got[0]
    return scan_order(ip), scan_order(comp)


def plain_scan(config):
    """``library_core.scan_library`` (the plain API) in ``scan_order``."""
    ip, comp = library_core.scan_library(config)
    return scan_order(ip), scan_order(comp)


def comparable_rows(rows):
    """Scan rows without ``mtime`` (a re-copied fixture's folder mtimes jitter by ~1 ms)."""
    tabs = [rows] if (not rows or isinstance(rows[0], dict)) else rows
    out = []
    for tab in tabs:
        out.append(sorted((json.dumps({k: v for k, v in norm(row).items() if k != "mtime"}, sort_keys=True)
                           for row in tab)))
    return out


def assert_same(results, label):
    """Legacy and new results agree; otherwise show a readable unified diff."""
    if results["legacy"] == results["new"]:
        return
    import difflib
    old = json.dumps(results["legacy"], indent=1, sort_keys=True, ensure_ascii=False, default=str).splitlines()
    new = json.dumps(results["new"], indent=1, sort_keys=True, ensure_ascii=False, default=str).splitlines()
    diff = "\n".join(list(difflib.unified_diff(old, new, "legacy", "new", lineterm="", n=2))[:120])
    pytest.fail(f"{label}: legacy and new differ\n{diff}")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    return Sandbox(tmp_path / "sb", monkeypatch)


def fresh(sandbox, builder, *args, **kwargs):
    built = sandbox.build(builder, *args, **kwargs)
    sandbox.reset()
    return built


def test_scans_and_merge_match_legacy_on_random_libraries(legacy, el, sandbox, qt_prompts):
    for state in range(FS_STATES):
        rng = random.Random(SEED * 31 + state)
        fresh(sandbox, build_library_fixture, rng, books=rng.randint(2, 8))
        config = {"translate_special_files": rng.random() < 0.3}
        results = run_both(sandbox, legacy, el,
                           lambda module, work: comparable_rows(scan_rows(module, config)))
        assert_same(results, f"state {state}")
        # the plain API (mobile) equals the desktop thread on the same tree
        sandbox.reset()
        plain = comparable_rows(library_core.scan_library(config))
        assert plain == results["new"]["out"], f"plain scan state {state}"
        assert sandbox.snapshot() == results["new"]["tree"]
        for scan in ("scan_output_folders", "scan_library_completed", "scan_for_epubs"):
            sandbox.reset()
            old_rows = comparable_rows(getattr(legacy, scan)(config))
            sandbox.reset()
            assert old_rows == comparable_rows(getattr(library_core, scan)(config)), (scan, state)
        sandbox.reset()
        output_rows = library_core.scan_output_folders(config)
        library_rows = library_core.scan_library_completed(config)
        in_progress, completed, _updates = library_core.merge_scans(output_rows, library_rows, config)
        assert comparable_rows((in_progress, completed)) == results["new"]["out"], f"merge state {state}"


@pytest.mark.parametrize("policy", ["keep_both", "replace", "skip"])
def test_organize_and_undo_match_legacy(legacy, el, sandbox, policy, qt_prompts):
    for state in range(max(4, FS_STATES // 4)):
        rng = random.Random(SEED * 7 + state)
        # a name collision in Library/Raw
        fresh(sandbox, build_library_fixture, rng, books=rng.randint(3, 7), extra=lambda built: (
            built["library"] / "Raw" / "Book00 Alpha.epub").write_bytes(b"collision"))
        config = {}

        def organize(module, work):
            ip, comp = scan_rows(module, config)
            MessageBoxStub.reset(answers=[MessageBoxStub.Yes])
            dialog = shelf_dialog(module, ip, comp, config, policy)
            module.EpubLibraryDialog._organize_into_library(dialog)
            return {"moves": dialog.files_reorganized.calls, "loads": dialog.loads}

        results = run_both(sandbox, legacy, el, organize)
        assert_same(results, f"organize state {state}")

        # plain API: plan + execute reproduces the desktop moves and summary
        work = sandbox.reset()
        ip, comp = plain_scan(config)
        shelf = library_core.LibraryShelf(ip, comp, config)
        plan = shelf.plan_organize()
        if plan["raw_moves"] or plan["translated_moves"]:
            result = shelf.execute_organize(plan, policy)
            assert norm(result["path_moves"]) == (results["new"]["out"]["moves"][0][0]
                                                  if results["new"]["out"]["moves"] else [])
            assert ["information", "Organize Files into Library", result["summary"]] in results["new"]["log"]
            assert sandbox.snapshot() == results["new"]["tree"]

        for choice in ("Raw", "Translated", "All"):
            def undo(module, work, choice=choice):
                ip, comp = scan_rows(module, config)
                MessageBoxStub.reset(answers=[MessageBoxStub.Yes])
                dialog = shelf_dialog(module, ip, comp, config, policy)
                module.EpubLibraryDialog._organize_into_library(dialog)
                ip, comp = scan_rows(module, config)
                MessageBoxStub.reset(answers=[choice])
                dialog = shelf_dialog(module, ip, comp, config, policy)
                module.EpubLibraryDialog._undo_organize_prompt(dialog)
                return {"moves": dialog.files_reorganized.calls, "loads": dialog.loads}

            results = run_both(sandbox, legacy, el, undo)
            assert_same(results, f"undo {choice} state {state}")


def test_delete_and_clear_raw_link_match_legacy(legacy, el, sandbox, qt_prompts):
    for state in range(max(4, FS_STATES // 3)):
        rng = random.Random(SEED * 11 + state)
        fresh(sandbox, build_library_fixture, rng, books=rng.randint(3, 7))
        config = {}
        picks = [rng.random() for _ in range(20)]
        keyword_answers = [rng.random() < 0.8 for _ in range(4)]

        def delete(module, work):
            ip, comp = scan_rows(module, config)
            chosen = [b for i, b in enumerate(ip + comp) if picks[i % len(picks)] < 0.5]
            dialog = shelf_dialog(module, ip, comp, config)
            started = []
            dialog._confirm_delete_simple = lambda targets: (
                MessageBoxStub.log.append(("simple", [t[1] for t in targets])) or list(targets))
            dialog._confirm_delete_with_keyword = lambda targets: (
                MessageBoxStub.log.append(("keyword", [t[1] for t in targets]))
                or ([t for i, t in enumerate(targets) if keyword_answers[i % 4]] or None))

            def start(targets, count, unregister):
                worker = module._LibraryDeleteThread(targets)
                results = []
                worker.delete_finished.connect(lambda r: results.extend(r))
                worker.run()
                started.append((sorted(results), count))
                module.EpubLibraryDialog._on_delete_finished(
                    dialog, worker, sorted(results), count, unregister, False)

            dialog._start_delete_worker = start
            module.EpubLibraryDialog._delete_books_prompt(dialog, chosen)
            return {"started": started, "loads": dialog.loads,
                    "selected": [sorted(dialog._selected_paths_ip), sorted(dialog._selected_paths_comp)]}

        results = run_both(sandbox, legacy, el, delete)
        assert_same(results, f"delete state {state}")

        # plain API on the same tree: plan_delete + execute_delete leave the same tree
        sandbox.reset()
        ip, comp = plain_scan(config)
        chosen = [b for i, b in enumerate(ip + comp) if picks[i % len(picks)] < 0.5]
        shelf = library_core.LibraryShelf(ip, comp, config)
        plan = shelf.plan_delete(chosen)
        if plan["targets"]:
            targets = (plan["targets"] if not plan["needs_keyword"] else
                       [t for i, t in enumerate(plan["targets"]) if keyword_answers[i % 4]])
            if targets:
                outcome = shelf.execute_delete(plan, targets)
                assert ["information" if not outcome["errors"] else "warning", "Delete",
                        outcome["summary"]] in results["new"]["log"]
                assert sandbox.snapshot() == results["new"]["tree"], f"plain delete state {state}"

        def clear(module, work):
            ip, comp = scan_rows(module, config)
            MessageBoxStub.reset(answers=[MessageBoxStub.Yes])
            dialog = shelf_dialog(module, ip, comp, config)
            flags = [module.EpubLibraryDialog._card_has_saved_raw_link(b) for b in ip + comp]
            module.EpubLibraryDialog._clear_saved_raw_link(dialog, ip + comp)
            return {"flags": flags, "loads": dialog.loads}

        results = run_both(sandbox, legacy, el, clear)
        assert_same(results, f"clear state {state}")


def test_import_and_output_override_match_legacy(legacy, el, sandbox, monkeypatch, qt_prompts):
    for state in range(max(4, FS_STATES // 4)):
        rng = random.Random(SEED * 13 + state)
        def extra(built):
            (built["downloads"] / "new raw.epub").write_bytes(b"PK\x03\x04 fake")
            (built["downloads"] / "notes.docx").write_bytes(b"x")

        fresh(sandbox, build_library_fixture, rng, books=rng.randint(2, 5), extra=extra)
        for target in ("raw", "translated"):
            for source in ("picker", "drop"):
                def importer(module, work, target=target, source=source):
                    dialog = shelf_dialog(module, [], [], {})
                    toasts = []
                    dialog._show_toast = lambda text: toasts.append(text)
                    paths = [str(work / "Downloads" / n) for n in ("new raw.epub", "notes.docx", "missing.epub")]
                    paths += [str(p) for p in sorted((work / "Downloads").glob("*.epub"))][:2]
                    module.EpubLibraryDialog._import_paths_into_library(dialog, paths, source=source, target=target)
                    return {"toasts": toasts, "loads": dialog.loads}

                results = run_both(sandbox, legacy, el, importer)
                assert_same(results, f"import {target}/{source} state {state}")
                # plain API (register in place, desktop semantics)
                work = sandbox.reset()
                paths = [str(work / "Downloads" / n) for n in ("new raw.epub", "notes.docx", "missing.epub")]
                paths += [str(p) for p in sorted((work / "Downloads").glob("*.epub"))][:2]
                library_core.import_paths(paths, target, {})
                assert sandbox.snapshot() == results["new"]["tree"]

        def override(module, work):
            ip, comp = scan_rows(module, {})
            config = {"output_directory": str(work / "Elsewhere")}
            dialog = shelf_dialog(module, ip, comp, config)
            persisted = []
            monkeypatch.setattr(module, "_persist_config_via_parent", lambda widget: persisted.append(1))
            dialog.parent = lambda: None
            MessageBoxStub.reset(answers=[MessageBoxStub.Yes])
            env_before = os.environ.get("OUTPUT_DIRECTORY")
            ok = module.EpubLibraryDialog._ensure_output_override_matches(dialog, ip + comp)
            result = {"ok": ok, "config": dict(config), "env": os.environ.get("OUTPUT_DIRECTORY"),
                      "persisted": persisted}
            if env_before is None:
                os.environ.pop("OUTPUT_DIRECTORY", None)
            else:
                os.environ["OUTPUT_DIRECTORY"] = env_before
            return result

        results = run_both(sandbox, legacy, el, override)
        assert_same(results, f"override state {state}")


def _raw_scan_books(books):
    """The rows ``_ScanForRawDialog.__init__`` keeps (workspaces missing their raw)."""
    return [dict(b) for b in (books or []) if bool(b.get("output_folder"))
            and (b.get("missing_raw_file") or not b.get("raw_source_path"))]


def test_scan_for_raw_matches_legacy(legacy, el, sandbox, qt_prompts):
    for state in range(max(4, FS_STATES // 4)):
        rng = random.Random(SEED * 17 + state)
        def extra(built):
            cands = built["downloads"] / "candidates"
            cands.mkdir()
            for index in range(8):
                name = rng.choice(["Book0{} Alpha", "book0{} beta", "Book0{} Gamma Translated",
                                   "zzz{}"]).format(index)
                ext = rng.choice([".epub", ".txt", ".pdf", ".html"])
                (cands / (name + ext)).write_bytes(b"raw")

        fresh(sandbox, build_library_fixture, rng, books=rng.randint(3, 7), extra=extra)
        for mode in ("exact", "fuzzy"):
            threshold = rng.choice([40, 70, 95])

            def scan(module, work, mode=mode, threshold=threshold):
                ip, comp = scan_rows(module, {})
                folder = str(work / "Downloads" / "candidates")
                books = _raw_scan_books(ip + comp)
                fake = fake_dialog(module._ScanForRawDialog, _books=books,
                                   _valid_exts={"epub", "txt", "pdf", "html"})
                fake._selected_exts = module._ScanForRawDialog._derive_auto_exts(fake) or set(fake._valid_exts)
                suffixes = module._ScanForRawDialog._ext_suffixes(fake)
                worker = module._RawScanWorker(folder, suffixes, module._LIBRARY_TRACKING_FILENAMES,
                                               books=books, mode=mode, threshold=threshold)
                got = []
                worker.results.connect(lambda f, candidates, matches: got.append((candidates, matches)))
                worker.run()
                candidates, matches = got[0]
                dialog = fake_dialog(module._ScanForRawDialog, _matches=matches, _candidates=candidates,
                                     _mode=mode, _threshold=threshold)
                dialog.applied = Emitter()
                dialog.accept = lambda: None
                status = []
                dialog._status_lbl = type("L", (), {"setText": lambda self, t: status.append(t)})()
                dialog._tree = type("T", (), {"blockSignals": lambda s, b: None, "clear": lambda s: None,
                                               "addTopLevelItem": lambda s, i: None})()
                dialog._apply_btn = type("A", (), {"setEnabled": lambda s, b: None})()

                from PySide6.QtWidgets import QTreeWidgetItem
                dialog._QTreeWidgetItem = QTreeWidgetItem
                module._ScanForRawDialog._populate_tree(dialog)
                module._ScanForRawDialog._apply_matches(dialog)
                return {"matches": sorted((k, v["path"], round(v["ratio"], 6), v["accepted"])
                                          for k, v in matches.items()),
                        "candidates": sorted(candidates), "status": status[-1],
                        "written": dialog.applied.calls[0][0]}

            results = run_both(sandbox, legacy, el, scan)
            assert_same(results, f"scan-for-raw {mode} state {state}")
            # plain API: RawScanSession over the shelf rows
            work = sandbox.reset()
            ip, comp = plain_scan({})
            session = library_core.RawScanSession(ip + comp, {"epub_library_scan_raw_mode": mode,
                                                              "epub_library_scan_raw_threshold": threshold})
            session.configure(folder=str(work / "Downloads" / "candidates"))
            outcome = session.scan()
            assert outcome["status"] == results["new"]["out"]["status"]
            assert sorted((k, v["path"], round(v["ratio"], 6), v["accepted"])
                          for k, v in outcome["matches"].items()) == [
                tuple(m) for m in results["new"]["out"]["matches"]]
            assert session.apply() == results["new"]["out"]["written"]
            assert sandbox.snapshot() == results["new"]["tree"]


class ReaderStub:
    calls: list = []

    def __init__(self, source, **kwargs):
        kwargs.pop("parent", None)
        kwargs.pop("config", None)
        provider = kwargs.pop("overlay_provider", None)
        kwargs["overlay_provider"] = norm(provider()) if provider else None
        ReaderStub.calls.append(("reader", source, norm(kwargs)))

    def __getattr__(self, name):
        return lambda *a, **k: None


class CursorStub:
    """Records the wait-cursor calls in order with the reader / overlay / system calls."""
    setOverrideCursor = staticmethod(lambda *a: ReaderStub.calls.append(("cursor-set",)))
    restoreOverrideCursor = staticmethod(lambda *a: ReaderStub.calls.append(("cursor-restore",)))
    processEvents = staticmethod(lambda *a: ReaderStub.calls.append(("process-events",)))


_TRACE_ONLY = {"cursor-set", "cursor-restore", "process-events", "overlay"}


def _desktop_reader_open(module, book, payload, initial, raw_only, config):
    """What Book Details' "open reader" does on the desktop: reader args or system viewer.

    The wait cursor, processEvents and the (expensive) overlay build are traced in call
    order, so the split into ``_plan_open_reader`` must keep the cursor up before the
    overlay is built, exactly as the pre-split method did.
    """
    ReaderStub.calls = []
    dialog = fake_dialog(module.BookDetailsDialog, _book=dict(book), _config=config,
                         _chapters_info=list(payload.get("chapters_info") or []),
                         _metadata_json=dict(payload.get("metadata_json") or {}),
                         _details=dict(payload.get("details") or {}),
                         _show_special_files=library_core._resolve_show_special_files(config))
    dialog._open_with_system_viewer = lambda target: ReaderStub.calls.append(("system", target))
    build_overlay = module.BookDetailsDialog._build_translated_overlay

    def traced_overlay():
        ReaderStub.calls.append(("overlay",))
        return build_overlay(dialog)

    dialog._build_translated_overlay = traced_overlay
    MessageBoxStub.reset()
    module.BookDetailsDialog._open_reader(dialog, initial, raw_only)
    calls = list(ReaderStub.calls)
    if any(entry[0] == "warning" for entry in MessageBoxStub.log):
        calls.append(("warning",))
    return calls


def test_book_details_loader_and_reader_plan_match_legacy(legacy, el, sandbox, monkeypatch, qt_prompts):
    for module in (legacy, el):
        monkeypatch.setattr(module, "EpubReaderDialog", ReaderStub)
        monkeypatch.setattr(module, "QApplication", CursorStub)
    for state in range(max(4, FS_STATES // 3)):
        rng = random.Random(SEED * 19 + state)
        fresh(sandbox, build_library_fixture, rng, books=rng.randint(2, 6))
        config = {"translate_special_files": rng.random() < 0.5}

        def details(module, work):
            ip, comp = scan_rows(module, config)
            out = []
            for book in ip + comp:
                loader = module._BookDetailsLoader(dict(book), config)
                got = {"preview": [], "done": [], "error": []}
                loader.preview_ready.connect(lambda p: got["preview"].append(p))
                loader.done.connect(lambda p: got["done"].append(p))
                loader.error.connect(lambda e: got["error"].append(e))
                loader.run()
                payload = got["done"][0] if got["done"] else {}
                plans = [_desktop_reader_open(module, book, payload, initial, raw_only, config)
                         for raw_only in (False, True) for initial in (None, 0, 1)]
                out.append({"book": book["path"], "got": got, "plans": plans})
            return out

        results = run_both(sandbox, legacy, el, details)
        assert_same(results, f"details state {state}")
        # plain API: load_book_details / plan_open_reader equal the desktop results
        sandbox.reset()
        ip, comp = plain_scan(config)
        for book, expected in zip(ip + comp, results["new"]["out"]):
            previews = []
            payload = library_core.load_book_details(dict(book), config, "full", on_preview=previews.append)
            assert norm(payload) == expected["got"]["done"][0]
            assert norm(previews) == expected["got"]["preview"]
            assert norm(library_core.load_book_details(dict(book), config, "preview")) == \
                expected["got"]["preview"][0]
            plans = []
            for raw_only in (False, True):
                for initial in (None, 0, 1):
                    plan = library_core.plan_open_reader(dict(book), payload, initial, raw_only, config)
                    ReaderStub.calls = []
                    if plan["mode"] in ("workspace", "epub"):
                        ReaderStub(plan["source"], **plan["kwargs"])
                        plans.append(list(ReaderStub.calls))
                    elif plan.get("target") and os.path.isfile(plan["target"]):
                        plans.append([("system", plan["target"])])
                    else:
                        plans.append([("warning",)])
            assert norm(plans) == [[call for call in desktop if call[0] not in _TRACE_ONLY]
                                   for desktop in expected["plans"]]
        # The traces really exercise the ordering: every overlay build happens under the
        # wait cursor (set before it, restored only after the reader is constructed).
        for book in results["new"]["out"]:
            for trace in book["plans"]:
                kinds = [call[0] for call in trace]
                if "overlay" in kinds:
                    assert "cursor-set" in kinds[:kinds.index("overlay")], trace
                    assert kinds.index("cursor-restore") > kinds.index("reader"), trace


def test_metadata_save_and_single_chapter_helpers_match_legacy(legacy, el, sandbox, monkeypatch, qt_prompts):
    class EditStub:
        Accepted = 1
        edits: dict = {}

        def __init__(self, values, parent=None):
            pass

        def exec(self):
            return 1

        def changed_values(self):
            return dict(EditStub.edits)

    for module in (legacy, el):
        monkeypatch.setattr(module, "_BookMetadataEditDialog", EditStub)
        monkeypatch.setattr(module, "QDialog", EditStub)
    for state in range(max(4, FS_STATES // 4)):
        rng = random.Random(SEED * 23 + state)
        fresh(sandbox, build_library_fixture, rng, books=rng.randint(2, 5))
        EditStub.edits = {"title": rng.choice(["New Title", ""]), "subject": rng.choice(["A, B", "#x"]),
                          "creator": "Someone"}

        def save(module, work):
            ip, comp = scan_rows(module, {})
            out = []
            for book in ip + comp:
                folder = book.get("output_folder") or ""
                dialog = fake_dialog(module.BookDetailsDialog, _book=dict(book), _metadata_json=dict(
                    book.get("metadata_json") or {}), _details={}, _config={},
                    _current_cover_path="", _chapters_info=[])
                dialog._apply_hero_payload = lambda payload: None
                dialog.parent = lambda: None
                MessageBoxStub.reset()
                module.BookDetailsDialog._on_edit_metadata_clicked(dialog)
                out.append({"log": list(MessageBoxStub.log), "metadata": dialog._metadata_json})
                if folder and os.path.isdir(folder):
                    out.append(module._mark_chapter_pending_for_retranslation(folder, "chapter0001.xhtml"))
                    out.append(module._cleanup_incomplete_chapter_output(folder, "chapter0002.xhtml"))
                    out.append(module._chapter_completed_in_progress(folder, "chapter0003.xhtml"))
            return out

        results = run_both(sandbox, legacy, el, save)
        assert_same(results, f"metadata state {state}")
        # plain API: save_metadata_json_atomic writes the same metadata.json
        sandbox.reset()
        ip, comp = plain_scan({})
        for book in ip + comp:
            try:
                library_core.save_metadata_json_atomic(dict(book), dict(EditStub.edits), {}, {})
            except library_core.MetadataEditError:
                pass
        plain_tree = {k: v for k, v in sandbox.snapshot().items() if k.endswith("metadata.json")}
        new_tree = {k: v for k, v in results["new"]["tree"].items() if k.endswith("metadata.json")}
        assert plain_tree == new_tree


# =============================================================================================
# 4. offscreen smoke: the desktop dialogs on a fixture library (legacy vs new)
# =============================================================================================

def test_library_and_book_details_dialogs_render_like_legacy(legacy, el, sandbox, monkeypatch):
    app = qapp()
    rng = random.Random(SEED * 29)
    sandbox.build(build_library_fixture, rng, books=7)
    from PySide6.QtCore import QEventLoop

    def wait_for(predicate, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents(QEventLoop.AllEvents, 50)
            if predicate():
                return True
            time.sleep(0.02)
        return False

    snapshots = {}
    for label, module in (("legacy", legacy), ("new", el)):
        sandbox.reset()
        monkeypatch.setattr(module, "_HAS_WEBENGINE", False)
        config = {"epub_library_card_size": "compact"}
        dialog = module.EpubLibraryDialog(config=config)
        dialog.show()  # the first scan starts from showEvent
        try:
            assert wait_for(lambda: bool(getattr(dialog, "_initial_scan_started", False))
                            and not (dialog._scanner_thread and dialog._scanner_thread.isRunning())
                            and (dialog._in_progress_books or dialog._completed_books))
            wait_for(lambda: False, timeout=0.6)
            dialog._auto_refresh_timer.stop()
            ip = sorted((norm({k: v for k, v in b.items() if k != "_cached_cover_path"})
                         for b in dialog._in_progress_books), key=lambda b: json.dumps(b, sort_keys=True))
            comp = sorted((norm({k: v for k, v in b.items() if k != "_cached_cover_path"})
                           for b in dialog._completed_books), key=lambda b: json.dumps(b, sort_keys=True))
            counts = dialog._ip_organize_btn.text(), dialog._comp_organize_btn.text(), \
                dialog._ip_undo_btn.text(), dialog._comp_undo_btn.text()
            filtered = [b["path"] for b in dialog._filtered(list(dialog._in_progress_books))]
            details = []
            for book in (dialog._in_progress_books + dialog._completed_books)[:3]:
                details_dialog = module.BookDetailsDialog(dict(book), config=config, parent=None)
                details_dialog.show()
                try:
                    assert wait_for(lambda: bool(details_dialog._chapters_info)
                                    or not (details_dialog._loader and details_dialog._loader.isRunning()),
                                    timeout=20)
                    wait_for(lambda: False, timeout=0.3)
                    details.append({
                        "rows": norm(details_dialog._chapters_info),
                        "toggle": details_dialog._toc_toggle.text(),
                        "strip": details_dialog._progress_strip.text()
                        if details_dialog._progress_strip.isVisibleTo(details_dialog) else None,
                        "editor": norm(details_dialog._metadata_editor_values()),
                    })
                finally:
                    details_dialog._auto_refresh_timer.stop() if hasattr(details_dialog, "_auto_refresh_timer") else None
                    details_dialog.close()
                    details_dialog.deleteLater()
            snapshots[label] = {"ip": ip, "comp": comp, "counts": counts, "filtered": filtered,
                                "details": details}
        finally:
            dialog.close()
            dialog.deleteLater()
            wait_for(lambda: False, timeout=0.3)
    assert snapshots["new"]["details"] and any(d["rows"] for d in snapshots["new"]["details"])
    assert snapshots["legacy"] == snapshots["new"]


# =============================================================================================
# 5. the plain API (Glossarion Mobile) agrees with the desktop decisions
# =============================================================================================

def test_library_env_seams(tmp_path, monkeypatch):
    import reader_doc

    output = tmp_path / "AppOutput"
    env = library_core.LibraryEnv(tmp_path / "AppLibrary", [output], tmp_path / "cache")
    monkeypatch.setattr(library_core, "_default_output_root", REAL_DEFAULT_OUTPUT_ROOT)
    try:
        library_core.install_library_env(env)
        assert os.environ["GLOSSARION_LIBRARY_DIR"] == str(tmp_path / "AppLibrary")
        assert library_core.get_library_dir() == str(tmp_path / "AppLibrary")
        assert library_core._default_output_root() == str(output)
        monkeypatch.delenv("OUTPUT_DIRECTORY", raising=False)
        assert library_core._resolve_output_roots({}) == [str(output)]
        assert library_covers._cover_cache_dir() == str(tmp_path / "cache" / "covers")
        assert reader_doc._epub_cache_dir() == str(tmp_path / "cache" / "epub")
        assert library_core.current_library_env() is env
    finally:
        library_core.uninstall_library_env()
    assert library_core.current_library_env() is None
    assert reader_doc._EPUB_CACHE_DIR_OVERRIDE is None and library_covers._COVER_CACHE_DIR_OVERRIDE is None


def test_mobile_import_copies_registers_and_scaffolds(tmp_path, monkeypatch):
    downloads = tmp_path / "picker-cache"
    epub = Path(write_epub(downloads / "My Book.epub", [("ch1.xhtml", "One", "<p>1</p>")], title="My Book"))
    text = downloads / "notes.txt"
    text.write_text("hello", encoding="utf-8")
    result = library_core.import_paths([str(epub), str(text), str(downloads / "x.docx")], "raw", {},
                                       copy_into_library=True, record_origins=True)
    raw_dir = Path(library_core.get_library_raw_dir())
    assert sorted(Path(p).name for p in result["imported"]) == ["My Book.epub", "notes.txt"]
    assert all(Path(p).parent == raw_dir for p in result["imported"])
    assert result["skipped"] and "x.docx" in result["skipped"][0]
    output = Path(os.environ["OUTPUT_DIRECTORY"])
    assert (output / "My Book" / "source_epub.txt").read_text(encoding="utf-8") == str(raw_dir / "My Book.epub")
    assert json.loads((output / "My Book" / "translation_progress.json").read_text(encoding="utf-8")) == {
        "chapters": {}, "chapter_chunks": {}, "version": "2.1"}
    assert str(raw_dir / "My Book.epub") in library_core.load_library_raw_inputs()
    assert library_core._load_origins()["raw"]["My Book.epub"] == str(epub.resolve())
    # identical content reuses the copy; different content keeps both
    again = library_core.import_paths([str(epub)], "raw", {}, copy_into_library=True)
    assert again["copied"][0]["reused"] is True
    other = downloads / "other" / "My Book.epub"
    other.parent.mkdir()
    other.write_bytes(b"different")
    third = library_core.import_paths([str(other)], "raw", {}, copy_into_library=True)
    assert Path(third["copied"][0]["path"]).name == "My Book (2).epub"
    ip, comp = library_core.scan_library({})
    assert any(b.get("translation_state") == "not_started" and b["folder_name"] == "My Book" for b in ip)
    # translations go to Library/Translated and show on the Completed shelf
    translated = library_core.import_paths([str(epub)], "translated", {}, copy_into_library=True)
    assert Path(translated["imported"][0]).parent == Path(library_core.get_library_translated_dir())
    ip, comp = library_core.scan_library({})
    assert any(b.get("in_library") for b in comp + ip)


def test_unique_destination_is_the_organize_rule(tmp_path):
    (tmp_path / "a.epub").write_bytes(b"1")
    (tmp_path / "a (2).epub").write_bytes(b"2")
    (tmp_path / "folder").mkdir()
    assert library_core.unique_destination(str(tmp_path), "a.epub") == str(tmp_path / "a (3).epub")
    assert library_core.unique_destination(str(tmp_path), "new.epub") == str(tmp_path / "new.epub")
    # the Organize rule only treats files as taken; include_dirs adds folders (FileBridge folders)
    assert library_core.unique_destination(str(tmp_path), "folder") == str(tmp_path / "folder")
    assert library_core.unique_destination(str(tmp_path), "folder", include_dirs=True) == str(tmp_path / "folder (2)")


def test_record_library_raw_inputs_is_the_run_setup_hook(tmp_path):
    book = tmp_path / "Book.epub"
    book.write_bytes(b"x")
    library_core.record_library_raw_inputs([str(book), str(tmp_path / "missing.epub"), "", None])
    assert library_core.load_library_raw_inputs() == [str(book)]


def test_library_shelf_plans_mirror_the_desktop_dialog(legacy, sandbox):
    rng = random.Random(SEED * 37)
    sandbox.build(build_library_fixture, rng, books=6)
    work = sandbox.reset()
    ip, comp = library_core.scan_library({})
    shelf = library_core.LibraryShelf(ip, comp, {})
    counts = shelf.counts()
    dialog = fake_dialog(legacy.EpubLibraryDialog, _in_progress_books=ip, _completed_books=comp, _config={})
    assert counts["raw_count"] == legacy.EpubLibraryDialog._count_raw_movable(dialog)
    assert counts["trans_count"] == legacy.EpubLibraryDialog._count_trans_movable(dialog)
    assert counts["missing_raw"] == sum(1 for b in ip + comp if b.get("missing_raw_file"))
    plan = shelf.plan_organize()
    assert {"raw_moves", "translated_moves", "preview", "collisions"} <= set(plan)
    delete_plan = shelf.plan_delete(ip[:2])
    assert set(delete_plan) == {"targets", "unregister", "needs_keyword", "detail", "simple_prompt"}
    assert library_core.is_delete_keyword(" Halgakos ") and library_core.is_delete_keyword("DELETE")
    assert not library_core.is_delete_keyword("remove")
    result = shelf.execute_delete(delete_plan)
    assert result["summary"].startswith("Deleted ")
    assert shelf.visible("comp") == shelf._filtered(list(comp))


def test_book_summary_keeps_the_card_semantics(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "response_ch1.html").write_text("x", encoding="utf-8")
    progress = {"chapters": {
        "1": {"status": "completed", "original_basename": "ch1.xhtml", "output_file": "response_ch1.html",
              "content_hash": "h1"},
        "2": {"status": "completed", "original_basename": "ch2.xhtml", "output_file": "response_ch2.html"},
        "3": {"status": "completed", "original_basename": "ch3.xhtml", "output_file": "response_ch1.html",
              "content_hash": "h3"},
        "gallery": {"status": "completed", "original_basename": "gallery.xhtml"},
        "meta": {"status": "completed", "original_basename": "source_epub.txt"},
    }, "chapter_chunks": {"h3": {"schema_version": 2, "total": 2, "chunks": {"1": "a", "2": "b"},
                                 "completed": [1], "entries": {
        "1": {"status": "completed"}, "2": {"status": "qa_failed", "qa_issues_found": ["X"]}}}}}
    (workspace / "translation_progress.json").write_text(json.dumps(progress), encoding="utf-8")
    summary = library_core.book_summary(str(workspace / "translation_progress.json"), {})
    # ch1 done; ch2 phantom (file gone) -> in progress; ch3 chunk QA failure -> failed
    assert (summary["total"], summary["completed"], summary["in_progress"], summary["failed"]) == (3, 1, 1, 1)
    assert summary == library_core._read_progress_summary(str(workspace / "translation_progress.json"),
                                                          exclude_special=True, config={})


def test_covers_resolve_without_qt(tmp_path):
    epub = write_epub(tmp_path / "c.epub", [("ch1.xhtml", "One", "<p>x</p>")], title="C")
    cover = library_covers.resolve_book_cover(epub, "epub", {})
    assert cover and open(cover, "rb").read().startswith(b"\x89PNG")
    assert library_covers.resolve_book_cover(epub, "epub", {}, should_stop=lambda: True) is library_covers.COVER_STOPPED
    folder = tmp_path / "ws"
    folder.mkdir()
    (folder / "cover.jpg").write_bytes(b"\xff\xd8\xff\xe0jpeg")
    assert library_covers.resolve_card_cover({"path": str(folder), "type": "in_progress"}) == str(folder / "cover.jpg")


def test_image_probes_agree_with_qt(tmp_path):
    qapp()
    pytest.importorskip("PIL")
    from PIL import Image
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QImage, QImageReader
    import io

    rng = random.Random(SEED * 41)
    samples = []
    decode_only = []
    for _ in range(90):
        width, height = rng.randint(1, 400), rng.randint(1, 400)
        fmt = rng.choice(["PNG", "JPEG", "GIF", "BMP", "WEBP", "TIFF", "PPM", "XBM", "PCX", "TGA", "ICO"])
        image = Image.new("RGB", (width, height), (rng.randint(0, 255), 10, 20))
        if fmt == "XBM":
            image = image.convert("1")
        for _k in range(12):
            image.putpixel((rng.randrange(width), rng.randrange(height)),
                           0 if fmt == "XBM" else (rng.randrange(256), 0, 0))
        buf = io.BytesIO()
        image.save(buf, fmt)
        data = buf.getvalue()
        # QImageReader misreports Pillow's TGA / multi-size ICO headers; sizes compared elsewhere
        (decode_only if fmt in ("TGA", "ICO") else samples).append(data)
        if fmt != "ICO":  # Qt decodes some cut ICO directories Pillow cannot open (DISCREPANCIES U5)
            for fraction in (0.97, 0.6, 0.3):
                decode_only.append(data[:max(1, int(len(data) * fraction))])
            for cut in (40, 54, 55, 56, 62, 66, 70, 72, 74, 80, 130):  # header / first-pixel edges
                decode_only.append(data[:cut])
    for width, height, extra in ((300, 300, ""), (100, 250, ""), ("300px", "240pt", ""), ("2in", "1in", ""),
                                 ("10mm", "1cm", ""), ("50%", "50%", 'viewBox="0 0 200 100"'),
                                 (None, None, 'viewBox="0 0 640 480"'), ("2in", None, 'viewBox="0 0 10 20"'),
                                 ("12.6", "7.4", ""), (None, None, 'viewBox="0,0,33.5,44.5"')):
        attrs = " ".join(f'{k}="{v}"' for k, v in (("width", width), ("height", height)) if v is not None)
        samples.append(f'<svg xmlns="http://www.w3.org/2000/svg" {attrs} {extra}><rect width="5" height="5"/></svg>'
                       .encode())
    samples += [b"", b"garbage", b"\x89PNG\r\n\x1a\n" + b"\x00" * 8]
    for data in samples:
        payload = QByteArray(data)
        buffer = QBuffer()
        buffer.setData(payload)
        buffer.open(QIODevice.ReadOnly)
        size = QImageReader(buffer).size()
        buffer.close()
        qt = (size.width(), size.height()) if size.isValid() else None
        ours = library_covers._probe_image_size(data)
        if qt is None:
            assert ours is None, (data[:20], ours)
        else:
            assert ours is not None and tuple(ours) == qt, (data[:40], qt, ours)
        assert library_covers._image_bytes_decodable(data) == (not QImage.fromData(data).isNull()), data[:16]
    for data in decode_only:
        assert library_covers._image_bytes_decodable(data) == (not QImage.fromData(data).isNull()), data[:16]
