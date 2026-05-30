#!/usr/bin/env python3
"""
Experimental PySide6 Android launcher for Glossarion.

This is a sidecar entry point for testing the desktop PySide6 GUI on Android.
It intentionally does not replace ../main.py, which remains the current
Buildozer/Kivy launcher.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import runpy
import sys
import tempfile
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
ANDROID_DIR = THIS_FILE.parents[1]
SRC_DIR = THIS_FILE.parents[2]
BACKEND_DIR = ANDROID_DIR / "_backend"
TRANSLATOR_GUI = SRC_DIR / "translator_gui.py"


def _prepend_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _is_android_runtime() -> bool:
    if sys.platform == "android":
        return True
    return any(
        os.environ.get(name)
        for name in (
            "ANDROID_ARGUMENT",
            "ANDROID_PRIVATE",
            "ANDROID_ROOT",
            "ANDROID_STORAGE",
        )
    )


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".glossarion_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _find_entrypoint(source_path: Path) -> Path:
    # python-for-android packages sources as top-level .pyc files.
    for candidate in (source_path, source_path.with_suffix(".pyc")):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot find desktop launcher as source or bytecode: {source_path}"
    )


def _configure_app_data_dir() -> None:
    """Point desktop GUI state at a writable Android data directory."""
    if not _is_android_runtime():
        return

    if os.environ.get("GLOSSARION_APP_DIR"):
        app_dir = Path(os.environ["GLOSSARION_APP_DIR"]).expanduser()
        if _is_writable_dir(app_dir):
            os.chdir(app_dir)
            return

    candidates = [
        os.environ.get("ANDROID_PRIVATE"),
        os.environ.get("ANDROID_ARGUMENT"),
        str(Path.home() / ".glossarion"),
        str(Path(tempfile.gettempdir()) / "glossarion"),
    ]

    for raw in candidates:
        if not raw:
            continue
        base = Path(raw).expanduser()
        candidate = base if base.name in {".glossarion", "glossarion"} else base / "glossarion"
        if _is_writable_dir(candidate):
            os.environ["GLOSSARION_APP_DIR"] = str(candidate)
            os.chdir(candidate)
            return


def _load_backend_stub(stub_name: str):
    """Load a stub from android/_backend without letting src/ shadow it."""
    stub_path = BACKEND_DIR / f"{stub_name}.py"
    if not stub_path.is_file():
        raise ImportError(f"Missing Android stub: {stub_path}")

    cached = sys.modules.get(stub_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(stub_name, stub_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Android stub: {stub_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[stub_name] = module
    spec.loader.exec_module(module)
    return module


def _register_android_stubs() -> None:
    """Pre-register modules that are unavailable or too heavy on Android."""
    if BACKEND_DIR.is_dir() and str(BACKEND_DIR) not in sys.path:
        sys.path.append(str(BACKEND_DIR))

    stub_map = {
        "tiktoken": ("tiktoken_stub", False),
        "ebooklib": ("ebooklib_stub", False),
        "ebooklib.epub": ("ebooklib_stub", False),
        "httpx": ("httpx_stub", False),
        "rapidfuzz": ("rapidfuzz_stub", False),
        "rapidfuzz.fuzz": ("rapidfuzz_stub", False),
        "rapidfuzz.process": ("rapidfuzz_stub", False),
        "langdetect": ("langdetect_stub", False),
        # The desktop implementations pull in native stacks that are not part
        # of the first PySide Android experiment.
        "image_translator": ("image_translator", True),
        "pdf_extractor": ("pdf_extractor", True),
    }

    for module_name, (stub_name, force_stub) in stub_map.items():
        if module_name in sys.modules:
            continue
        if not force_stub:
            try:
                importlib.import_module(module_name)
                continue
            except ImportError:
                pass
        try:
            stub = _load_backend_stub(stub_name)
            sys.modules[module_name] = stub
        except Exception:
            # Keep startup permissive. The desktop GUI will surface the real
            # import error if a feature needs the missing module.
            pass


def _prepare_runtime() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("GLOSSARION_ANDROID_PYSIDE", "1")

    # Keep src/ ahead of _backend/ so translator_gui.py can import its real
    # PySide mixins. Individual heavy modules are stubbed explicitly above.
    _prepend_path(SRC_DIR)
    if BACKEND_DIR.is_dir() and str(BACKEND_DIR) not in sys.path:
        sys.path.append(str(BACKEND_DIR))

    _configure_app_data_dir()
    _register_android_stubs()


def main() -> None:
    _prepare_runtime()

    entrypoint = _find_entrypoint(TRANSLATOR_GUI)
    sys.argv = [str(entrypoint), *sys.argv[1:]]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
