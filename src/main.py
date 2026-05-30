#!/usr/bin/env python3
"""PySide6 Android entrypoint for Glossarion.

PySide's Android deploy tool requires a file named main.py in the project
root. Keep the real launcher logic in android/pyside_launcher/main.py so it
does not collide with the existing Kivy Android entrypoint.
"""

from __future__ import annotations

import runpy
import sys
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


if __name__ == "__main__":
    launcher = _find_entrypoint(LAUNCHER)
    sys.argv = [str(launcher), *sys.argv[1:]]
    runpy.run_path(str(launcher), run_name="__main__")
