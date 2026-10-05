"""Glossarion mobile entry point (Flet app module, ``[tool.flet.app] module = "main"``).

``runtime_bootstrap.bootstrap()`` runs before anything else: it applies the
backend environment contract, chdirs into the app data dir, puts the backend on
``sys.path`` and installs logging/crash hooks. Only then is Flet imported and
the UI started. No backend module is imported here; the UI warm-imports them
on a worker thread once the page is up.

``main`` (``glossarion_mobile.app``) builds the chat-first shell: route ``/`` is
the chat home, ``/__selftest__`` runs the self-test (CI contract with
``ci/android_smoke.sh``: ``GLOSSARION_READY`` -> ``GLOSSARION_BACKEND_READY`` ->
``GLOSSARION_SELFTEST PASS``), and the U0 device checks open from Settings >
Logs & diagnostics.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from glossarion_mobile import runtime_bootstrap  # noqa: E402  (stdlib-only module)

PATHS = runtime_bootstrap.bootstrap(app_dir=APP_DIR)

import flet as ft  # noqa: E402

from glossarion_mobile.app import main  # noqa: E402

__all__ = ["main", "PATHS"]

if __name__ == "__main__":
    # Absolute assets dir: bootstrap() changed the cwd to the data dir. (On a
    # device Flet's FLET_ASSETS_DIR override points at <app>/assets anyway.)
    ft.run(main, assets_dir=str(APP_DIR / "assets"))
