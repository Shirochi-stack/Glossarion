"""An EPUB under an unsupported extension is renamed to .epub (translation_pipeline._windows_supported_input_path).

User report: a picked EPUB arrived as the Windows short name ``61B035~1.EPU`` and the run rejected
it as an unsupported file type. The input rename every selection and run goes through
(``_normalize_windows_input_filenames`` -> ``_windows_supported_input_path``) now expands a short
name first (never renaming it) and renames an EPUB whose extension is not a supported input type.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_epub_extension_detect.py
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _tp():
    """translation_pipeline, imported when a test runs (a collection-time import changes the
    module state other files in the python-app list freeze their goldens against)."""
    import translation_pipeline

    return translation_pipeline


class Owner:
    """The two moved input helpers on a minimal owner (log, config, output base)."""

    def _windows_supported_input_path(self, path):
        return _tp().GlossaryPipelineMixin._windows_supported_input_path(self, path)

    def _normalize_windows_input_filenames(self, paths):
        return _tp().GlossaryPipelineMixin._normalize_windows_input_filenames(self, paths)

    def __init__(self, output_dir: str = "") -> None:
        self.logs: list[str] = []
        self.config: dict = {"output_directory": output_dir} if output_dir else {}
        self.saved = 0

    def append_log(self, message: str) -> None:
        self.logs.append(message)

    def save_config(self, show_message: bool = False) -> None:
        self.saved += 1

    def _remap_windows_renamed_epub_glossary(self, old, new) -> None:
        pass


def write_epub(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", "<container/>")
        zf.writestr("OEBPS/content.opf", "<package/>")
    return str(path)


@pytest.fixture(autouse=True)
def _no_short_names(monkeypatch):
    monkeypatch.setattr(_tp(), "_windows_long_path", lambda path: path)


@pytest.mark.parametrize("name, expected", [
    ("61B035~1.EPU", "61B035~1.epub"),
    ("Book.EPU", "Book.epub"),
    ("Book.download", "Book.epub"),
    ("Book", "Book.epub"),
])
def test_an_epub_under_an_unsupported_extension_is_renamed(tmp_path, name, expected):
    source = write_epub(tmp_path / name)
    owner = Owner()
    result = owner._windows_supported_input_path(source)
    assert result == str(tmp_path / expected)
    # listdir, not exists(): Windows may give the new name the old 8.3 alias
    assert os.listdir(tmp_path) == [expected]
    assert any("is an EPUB; renamed it" in line for line in owner.logs)


def test_a_zip_epub_is_left_to_the_zip_route(tmp_path):
    source = write_epub(tmp_path / "Book.zip")
    assert Owner()._windows_supported_input_path(source) == source
    # the existing ZIP resolution copies it to Book.epub
    from input_preparation import resolve_input_to_epub

    resolved = resolve_input_to_epub(source, should_stop=lambda: False, log=lambda _m: None)
    assert resolved == str(tmp_path / "Book.epub") and os.path.isfile(source)


def test_a_zip_that_is_not_an_epub_is_left_alone(tmp_path):
    source = tmp_path / "notes.EPU"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("readme.txt", "hello")
    plain = tmp_path / "plain"
    plain.write_bytes(b"not a zip at all")
    owner = Owner()
    assert owner._windows_supported_input_path(str(source)) == str(source)
    assert owner._windows_supported_input_path(str(plain)) == str(plain)
    assert os.path.isfile(source) and os.path.isfile(plain) and not owner.logs


def test_supported_extensions_are_never_content_checked(tmp_path, monkeypatch):
    import image_archive_epub

    calls = []
    monkeypatch.setattr(image_archive_epub, "is_epub_zip", lambda p: calls.append(p) or True)
    source = tmp_path / "chapter.txt"
    source.write_text("text", encoding="utf-8")
    assert Owner()._windows_supported_input_path(str(source)) == str(source)
    assert calls == []


def test_an_existing_target_is_never_replaced(tmp_path):
    taken = tmp_path / "Book.epub"
    taken.write_bytes(b"another book")
    source = write_epub(tmp_path / "Book.EPU")
    result = Owner()._windows_supported_input_path(source)
    assert result == str(tmp_path / "Book_windows_safe_2.epub")
    assert taken.read_bytes() == b"another book"


def test_a_failed_rename_copies_into_the_output_folder(tmp_path, monkeypatch):
    source = write_epub(tmp_path / "in" / "Book.EPU")
    out = tmp_path / "out"

    def refuse(*_args):
        raise PermissionError("read-only folder")

    monkeypatch.setattr(_tp().os, "rename", refuse)
    owner = Owner(output_dir=str(out))
    result = owner._windows_supported_input_path(source)
    assert result == str(out / "Book.epub") and os.path.isfile(result) and os.path.isfile(source)
    # picking it again reuses that copy
    assert owner._windows_supported_input_path(source) == result
    assert not (out / "Book_windows_safe_2.epub").exists()


def test_a_short_name_is_expanded_and_never_renamed(tmp_path, monkeypatch):
    real = write_epub(tmp_path / "61b035f91d56ed3b328f_Lets raise my cousin raw.epub")
    short = str(tmp_path / "61B035~1.EPU")  # what the picker handed over
    monkeypatch.setattr(_tp(), "_windows_long_path", lambda p: real if p == short else p)
    owner = Owner()
    assert owner._windows_supported_input_path(short) == real
    assert sorted(os.listdir(tmp_path)) == [os.path.basename(real)]
    assert any("full file name" in line for line in owner.logs)


def test_an_unusable_long_name_is_copied_under_that_name(tmp_path, monkeypatch):
    short = write_epub(tmp_path / "in" / "61B035~1.EPU")
    long_name = str(tmp_path / "in" / "Lets raise my cousin raw.epub")  # e.g. beyond MAX_PATH
    monkeypatch.setattr(_tp(), "_windows_long_path", lambda p: long_name if p == short else p)
    out = tmp_path / "out"
    result = Owner(output_dir=str(out))._windows_supported_input_path(short)
    assert result == str(out / "Lets raise my cousin raw.epub")
    assert os.path.isfile(short)  # the short name itself is never renamed


def test_selection_normalization_saves_the_new_path(tmp_path):
    source = write_epub(tmp_path / "Book.EPU")
    owner = Owner()
    assert owner._normalize_windows_input_filenames([source]) == [str(tmp_path / "Book.epub")]
    assert owner.config["last_input_files"] == [str(tmp_path / "Book.epub")] and owner.saved == 1


def test_windows_long_path_is_a_no_op_off_windows_or_without_a_tilde(monkeypatch):
    monkeypatch.undo()
    real = _tp()._windows_long_path
    assert real("C:/Books/Novel.epub") == "C:/Books/Novel.epub"
    monkeypatch.setattr(_tp().sys, "platform", "linux")
    assert real("/books/NOVEL~1.EPU") == "/books/NOVEL~1.EPU"


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="GetLongPathNameW is Windows only")
def test_windows_long_path_expands_a_real_short_name(tmp_path, monkeypatch):
    monkeypatch.undo()
    import ctypes

    real = write_epub(tmp_path / "Lets raise my cousin raw.epub")
    buffer = ctypes.create_unicode_buffer(1024)
    if not ctypes.windll.kernel32.GetShortPathNameW(real, buffer, 1024) or "~" not in buffer.value:
        pytest.skip("8.3 names are disabled on this volume")
    assert os.path.normcase(_tp()._windows_long_path(buffer.value)) == os.path.normcase(real)
