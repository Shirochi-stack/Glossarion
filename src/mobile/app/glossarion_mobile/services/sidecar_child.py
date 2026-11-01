"""What a child interpreter of Generate MD / TXT runs (``job_kinds.compile.sidecars_in_children``)."""

from __future__ import annotations

import contextlib
import io

from bs4 import FeatureNotFound

import enhanced_text_extractor  # noqa: F401  (loaded when the child starts: the first file should not pay for it)
import md_txt_sidecar_writer

__all__ = ["NEEDS_LXML", "write"]

#: ``write``'s answer for a chapter the extractor wants lxml for: do it on a thread
NEEDS_LXML = "needs lxml"


def write(args: tuple):
    """``_write_html_md_txt_sidecars(output_dir, output_filename, target_html, **options)``. The extractor prints that it
    failed before it raises ``FeatureNotFound``: that line is not true of a chapter that then converts on a thread, so
    it stays here."""
    output_dir, output_filename, target_html, options = args
    printed = io.StringIO()
    keep = True
    try:
        with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
            return md_txt_sidecar_writer._write_html_md_txt_sidecars(output_dir, output_filename, target_html, **options)
    except FeatureNotFound:
        keep = False
        return NEEDS_LXML
    finally:
        if keep and printed.getvalue():
            print(printed.getvalue(), end="")
