#!/usr/bin/env python3
"""PySide6 Android entrypoint for Glossarion.

PySide's Android deploy tool requires a file named main.py in the project
root. Keep the real launcher logic in android/pyside_launcher/main.py so it
does not collide with the existing Kivy Android entrypoint.
"""

from __future__ import annotations

import os
import runpy
import sys
import tempfile
import traceback
from pathlib import Path


LAUNCHER = Path(__file__).resolve().parent / "android" / "pyside_launcher" / "main.py"


def _find_entrypoint(source_path: Path) -> Path:
    # python-for-android packages sources as top-level .pyc files.
    for candidate in (source_path, source_path.with_suffix(".pyc")):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot find Android launcher as source or bytecode: {source_path}"
    )


def _write_startup_crash(exc: BaseException) -> None:
    message = (
        "Glossarion PySide Android root launcher failed\n\n"
        + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    )
    try:
        print(message, file=sys.stderr, flush=True)
    except Exception:
        pass
    for raw in (
        os.environ.get("ANDROID_PRIVATE"),
        os.environ.get("ANDROID_ARGUMENT"),
        tempfile.gettempdir(),
    ):
        if not raw:
            continue
        try:
            base = Path(raw).expanduser()
            base.mkdir(parents=True, exist_ok=True)
            (base / "glossarion_pyside_startup_crash.log").write_text(
                message,
                encoding="utf-8",
            )
            return
        except Exception:
            pass


if __name__ == "__main__":
    try:
        launcher = _find_entrypoint(LAUNCHER)
        sys.argv = [str(launcher), *sys.argv[1:]]
        runpy.run_path(str(launcher), run_name="__main__")
    except BaseException as exc:
        _write_startup_crash(exc)
        raise
