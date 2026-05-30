#!/usr/bin/env python3
"""PySide6 Android entrypoint for Glossarion.

PySide's Android deploy tool requires a file named main.py in the project
root. Keep the real launcher logic in android/pyside_launcher/main.py so it
does not collide with the existing Kivy Android entrypoint.
"""

from __future__ import annotations

import runpy
from pathlib import Path


LAUNCHER = Path(__file__).resolve().parent / "android" / "pyside_launcher" / "main.py"


if __name__ == "__main__":
    runpy.run_path(str(LAUNCHER), run_name="__main__")
