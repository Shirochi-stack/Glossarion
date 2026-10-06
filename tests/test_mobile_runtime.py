"""Unit tests for src/mobile_runtime.py (runtime gates shared by desktop and Glossarion Mobile).

The desktop never sets GLOSSARION_MOBILE / GLOSSARION_NO_PROCESSES / FLET_PLATFORM /
GLOSSARION_DATA_DIR / CONFIG_FILE, so with those unset every gate must return the
desktop behaviour: process pools allowed and paths returned unchanged.

Python 3.10 compatible (python-app.yml runs this on 3.10). Run:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_mobile_runtime.py
"""

import ast
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:  # CI has no tests/conftest.py
    sys.path.insert(0, str(SRC_DIR))

import mobile_runtime  # noqa: E402

GATE_ENV = ("GLOSSARION_MOBILE", "GLOSSARION_NO_PROCESSES", "FLET_PLATFORM", "GLOSSARION_DATA_DIR", "CONFIG_FILE")
TRUTHY = ("1", "true", "TRUE", "yes", "on", " 1 ", "On")
FALSY = ("", "0", "false", "no", "off", "2", "enabled")


@pytest.fixture
def desktop(monkeypatch):
    """A desktop process: none of the mobile env vars, a desktop sys.platform."""
    for name in GATE_ENV:
        monkeypatch.delenv(name, raising=False)
    if sys.platform in ("android", "ios", "emscripten", "wasi"):  # pragma: no cover - host tests only
        monkeypatch.setattr(sys, "platform", "linux")
    return monkeypatch


# --------------------------------------------------------------------------
# Desktop defaults (env unset)
# --------------------------------------------------------------------------


def test_desktop_defaults_with_env_unset(desktop):
    assert mobile_runtime.is_mobile() is False
    assert mobile_runtime.processes_available() is True
    assert mobile_runtime.subprocesses_available() is True


def test_data_dir_returns_the_default_object_unchanged(desktop, tmp_path):
    default = str(tmp_path / "app dir")
    assert mobile_runtime.data_dir(default) is default
    path_default = tmp_path / "as-path"
    assert mobile_runtime.data_dir(path_default) is path_default
    assert mobile_runtime.data_dir(None) is None


def test_config_file_default_is_config_json_in_the_default_dir(desktop, tmp_path):
    assert mobile_runtime.config_file(str(tmp_path)) == os.path.join(str(tmp_path), "config.json")


def test_desktop_sys_platforms_allow_processes(desktop):
    for platform_name in ("win32", "linux", "darwin", "cygwin", "freebsd14"):
        desktop.setattr(sys, "platform", platform_name)
        assert mobile_runtime.is_mobile() is False, platform_name
        assert mobile_runtime.processes_available() is True, platform_name


# --------------------------------------------------------------------------
# GLOSSARION_MOBILE / GLOSSARION_NO_PROCESSES / FLET_PLATFORM / sys.platform
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", TRUTHY)
def test_glossarion_mobile_truthy_enables_mobile_and_disables_processes(desktop, value):
    desktop.setenv("GLOSSARION_MOBILE", value)
    assert mobile_runtime.is_mobile() is True
    assert mobile_runtime.processes_available() is False
    assert mobile_runtime.subprocesses_available() is False


@pytest.mark.parametrize("value", FALSY)
def test_glossarion_mobile_falsy_keeps_desktop(desktop, value):
    desktop.setenv("GLOSSARION_MOBILE", value)
    assert mobile_runtime.is_mobile() is False
    assert mobile_runtime.processes_available() is True


@pytest.mark.parametrize("value", TRUTHY)
def test_no_processes_flag_disables_pools_without_claiming_mobile(desktop, value):
    desktop.setenv("GLOSSARION_NO_PROCESSES", value)
    assert mobile_runtime.is_mobile() is False
    assert mobile_runtime.processes_available() is False
    assert mobile_runtime.subprocesses_available() is False


@pytest.mark.parametrize("value", FALSY)
def test_no_processes_falsy_keeps_pools(desktop, value):
    desktop.setenv("GLOSSARION_NO_PROCESSES", value)
    assert mobile_runtime.processes_available() is True


@pytest.mark.parametrize("value", ("android", "ios", "Android", " IOS "))
def test_flet_platform_mobile_values(desktop, value):
    desktop.setenv("FLET_PLATFORM", value)
    assert mobile_runtime.is_mobile() is True
    assert mobile_runtime.processes_available() is False


@pytest.mark.parametrize("value", ("windows", "macos", "linux", "web", "", "fuchsia"))
def test_flet_platform_desktop_values(desktop, value):
    desktop.setenv("FLET_PLATFORM", value)
    assert mobile_runtime.is_mobile() is False
    assert mobile_runtime.processes_available() is True


@pytest.mark.parametrize("platform_name", ("android", "ios"))
def test_mobile_sys_platform(desktop, platform_name):
    desktop.setattr(sys, "platform", platform_name)
    assert mobile_runtime.is_mobile() is True
    assert mobile_runtime.processes_available() is False


@pytest.mark.parametrize("platform_name", ("emscripten", "wasi"))
def test_wasm_platforms_have_no_processes_but_are_not_mobile(desktop, platform_name):
    desktop.setattr(sys, "platform", platform_name)
    assert mobile_runtime.is_mobile() is False
    assert mobile_runtime.processes_available() is False


def test_gates_are_read_at_call_time(desktop):
    assert mobile_runtime.processes_available() is True
    desktop.setenv("GLOSSARION_NO_PROCESSES", "1")
    assert mobile_runtime.processes_available() is False
    desktop.delenv("GLOSSARION_NO_PROCESSES")
    assert mobile_runtime.processes_available() is True


# --------------------------------------------------------------------------
# data_dir / config_file
# --------------------------------------------------------------------------


def test_data_dir_override(desktop, tmp_path):
    override = str(tmp_path / "mobile-data")
    desktop.setenv("GLOSSARION_DATA_DIR", "  " + override + "  ")
    assert mobile_runtime.data_dir("/desktop/app") == override


def test_blank_data_dir_override_is_ignored(desktop):
    desktop.setenv("GLOSSARION_DATA_DIR", "   ")
    default = "/desktop/app"
    assert mobile_runtime.data_dir(default) is default


def test_config_file_env_wins(desktop, tmp_path):
    explicit = str(tmp_path / "custom" / "my-config.json")
    desktop.setenv("CONFIG_FILE", " " + explicit + " ")
    desktop.setenv("GLOSSARION_DATA_DIR", str(tmp_path / "data"))
    assert mobile_runtime.config_file("/desktop/app") == explicit


def test_config_file_follows_data_dir(desktop, tmp_path):
    data = str(tmp_path / "data")
    desktop.setenv("GLOSSARION_DATA_DIR", data)
    assert mobile_runtime.config_file("/desktop/app") == os.path.join(data, "config.json")


def test_blank_config_file_env_is_ignored(desktop, tmp_path):
    desktop.setenv("CONFIG_FILE", "  ")
    assert mobile_runtime.config_file(str(tmp_path)) == os.path.join(str(tmp_path), "config.json")


# --------------------------------------------------------------------------
# make_pool_executor
# --------------------------------------------------------------------------


_INIT_CALLS = []


def _record_initializer(*args):
    _INIT_CALLS.append(args)


def test_desktop_pool_is_a_real_process_pool(desktop):
    executor = mobile_runtime.make_pool_executor(2)
    try:
        assert type(executor) is ProcessPoolExecutor
        assert executor._max_workers == 2
        assert executor.submit(os.getpid).result(timeout=120) != os.getpid()
    finally:
        executor.shutdown(wait=True)


def test_desktop_pool_passes_initializer_and_initargs(desktop):
    executor = mobile_runtime.make_pool_executor(3, initializer=_record_initializer, initargs=["a", 1])
    try:
        assert type(executor) is ProcessPoolExecutor
        assert executor._max_workers == 3
        assert executor._initializer is _record_initializer
        assert executor._initargs == ("a", 1)
    finally:
        executor.shutdown(wait=True)


def test_desktop_pool_without_initializer(desktop):
    executor = mobile_runtime.make_pool_executor(1)
    try:
        assert executor._initializer is None
        assert executor._initargs == ()
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("env_name,value", (("GLOSSARION_NO_PROCESSES", "1"), ("GLOSSARION_MOBILE", "1"),
                                            ("FLET_PLATFORM", "android")))
def test_thread_mode_never_runs_the_initializer(desktop, env_name, value):
    desktop.setenv(env_name, value)
    del _INIT_CALLS[:]
    executor = mobile_runtime.make_pool_executor(2, initializer=_record_initializer, initargs=("x",))
    try:
        assert type(executor) is ThreadPoolExecutor
        assert executor._max_workers == 2
        results = [executor.submit(threading.current_thread).result(timeout=30) for _ in range(4)]
        assert executor.submit(os.getpid).result(timeout=30) == os.getpid()
    finally:
        executor.shutdown(wait=True)
    assert _INIT_CALLS == []
    assert all(t is not threading.main_thread() for t in results)
    assert all(t.name.startswith("glossarion-pool") for t in results)


def test_thread_mode_uses_the_given_thread_name_prefix(desktop):
    desktop.setenv("GLOSSARION_NO_PROCESSES", "1")
    with mobile_runtime.make_pool_executor(1, thread_name_prefix="chapter-extract") as executor:
        name = executor.submit(lambda: threading.current_thread().name).result(timeout=30)
    assert name.startswith("chapter-extract")


@pytest.mark.parametrize("requested,expected", ((0, 1), (None, 1), (-4, 1), ("3", 3), (5, 5), (2.9, 2)))
def test_worker_count_is_clamped_to_at_least_one(desktop, requested, expected):
    desktop.setenv("GLOSSARION_NO_PROCESSES", "1")
    executor = mobile_runtime.make_pool_executor(requested)
    try:
        assert executor._max_workers == expected
    finally:
        executor.shutdown(wait=True)


def test_desktop_worker_count_is_clamped_too(desktop):
    executor = mobile_runtime.make_pool_executor(0)
    try:
        assert type(executor) is ProcessPoolExecutor
        assert executor._max_workers == 1
    finally:
        executor.shutdown(wait=True)


# --------------------------------------------------------------------------
# Module hygiene
# --------------------------------------------------------------------------


def _source():
    return (SRC_DIR / "mobile_runtime.py").read_text(encoding="utf-8-sig")


def test_python_310_syntax():
    ast.parse(_source(), feature_version=(3, 10))


def test_stdlib_only_imports():
    tree = ast.parse(_source())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"os", "sys", "concurrent"}, imported


def test_public_api():
    for name in ("is_mobile", "processes_available", "subprocesses_available", "data_dir", "config_file",
                 "make_pool_executor"):
        assert callable(getattr(mobile_runtime, name)), name


# --------------------------------------------------------------------------
# Desktop packaging: every PyInstaller spec ships the U1-U7 shared modules
# --------------------------------------------------------------------------

# Imported by core desktop modules (TransateKRtoEN, translator_gui, epub_converter, the PDF
# and QA paths, bubble_detector, ...), so every tier needs them in both lists.
U1_SHARED_MODULES = (
    "mobile_runtime", "app_paths", "config_store", "prompt_defaults", "metadata_defaults",
    "ollama_settings", "key_pools", "output_naming", "pdf_mupdf_html",
)
# U2: TranslatorGUI inherits the owner_state/run_env/settings_persistence mixins and its Direct
# Text dialog uses headless_owner.DirectTextRunOptions; the settings schema ships with them.
U2_SHARED_MODULES = (
    "owner_state", "run_env", "settings_persistence", "headless_owner", "settings_schema",
    "settings_schema_data",
    # run_env's EPUB compile env resolves the source EPUB through it (epub_library re-exports it)
    "library_core",
)

# U3: TranslatorGUI inherits the text-job / input-preparation / pipeline mixins, its Stop button
# and run start use stop_control, and the Direct Text dialog inherits the chat store and stream
# mixins (translator_gui imports all of them at module level).
U3_SHARED_MODULES = (
    "stop_control", "job_runner", "text_jobs", "input_preparation", "translation_pipeline",
    "direct_text_store", "direct_text_stream",
)

# U4: translator_gui (model catalog, route controls), other_settings (prompt profiles, locks),
# GlossaryManager_GUI (mode locks), multi_api_key_manager / unified_api_client (key pools,
# refusal defaults) and authgem/authcd/authgrok (the shared OAuth session) import these.
U4_SHARED_MODULES = (
    "settings_rules", "model_catalog_core", "prompt_profiles", "key_pool_service", "oauth_session",
)

# U5: epub_library re-imports the Library cover / reader document / live stream moves,
# Retranslation_GUI inherits progress_core.ProgressViewMixin and binds the progress_actions /
# glossary_progress_core closures, and TransateKRtoEN's cleanup_missing_files delegates to
# progress_core (all at module level). They are GUI-free, so every tier ships them, including
# the Lite builds that leave epub_library (QtWebEngine, ~152 MB) out on purpose.
U5_SHARED_MODULES = (
    "library_covers", "reader_doc", "live_stream", "progress_core", "progress_actions",
    "glossary_progress_core",
)
#: Specs that deliberately do not bundle epub_library (it drags in Chromium WebEngine).
LITE_WITHOUT_EPUB_LIBRARY = ("translator_lite.spec", "translator_TurboLite.spec", "translator_linux_TurboLite.spec")

# U6: GlossaryManager_GUI keeps thin wrappers over glossary_document (the Glossary Editor and the
# prompt-profile config helpers), translator_gui over glossary_files (editor backups, delete /
# restore glossary files, Map Glossaries, JSON repair) and parallel_epub_glossary re-exports
# parallel_epub_core (all imported at module level), so every tier ships them.
U6_SHARED_MODULES = ("glossary_document", "glossary_files", "parallel_epub_core")
#: Desktop module -> the U6 core it imports at module level.
U6_IMPORTERS = {
    "GlossaryManager_GUI": "glossary_document",
    "translator_gui": "glossary_files",
    "parallel_epub_glossary": "parallel_epub_core",
}

# U7: translation_pipeline inherits the image / generative and RPG Maker runner mixins,
# async_api_processor is a thin Qt view over async_batch_core, other_settings calls the output
# tools core (Load Font, Validate EPUB, Delete Header Files / TOC.txt) and Retranslation_GUI /
# progress_actions import the SDLXLIFF reviewer core (all at module level), so every tier ships them.
U7_SHARED_MODULES = ("image_job", "rpgmaker_job", "async_batch_core", "output_tools_core", "sdlxliff_review_core")
#: Desktop or shared module -> the U7 cores it imports at module level.
U7_IMPORTERS = {
    "translation_pipeline": ("image_job", "rpgmaker_job"),
    "async_api_processor": ("async_batch_core",),
    "other_settings": ("output_tools_core",),
    "Retranslation_GUI": ("sdlxliff_review_core",),
    "progress_actions": ("sdlxliff_review_core",),
}


def _spec_list(source, name):
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{name} = ["))
    end = next(j for j in range(start + 1, len(lines)) if lines[j].rstrip() == "]")
    return ast.literal_eval("\n".join(lines[start:end + 1]).split("=", 1)[1].strip())


def test_u1_shared_modules_are_packaged_in_every_spec():
    specs = sorted(SRC_DIR.glob("translator*.spec"))
    assert len(specs) == 14, [p.name for p in specs]
    for spec in specs:
        source = spec.read_text(encoding="utf-8")
        files = [Path(entry[0]).stem for entry in _spec_list(source, "app_files")]
        modules = _spec_list(source, "app_modules")
        for name in (U1_SHARED_MODULES + U2_SHARED_MODULES + U3_SHARED_MODULES + U4_SHARED_MODULES
                     + U5_SHARED_MODULES + U6_SHARED_MODULES + U7_SHARED_MODULES):
            assert (SRC_DIR / f"{name}.py").is_file(), name
            assert files.count(name) == 1, (spec.name, name, "app_files")
            assert modules.count(name) == 1, (spec.name, name, "app_modules")


def test_u5_modules_ship_where_epub_library_is_left_out():
    """The Lite tiers keep epub_library commented out; the U5 cores never need it at import time."""
    for spec in sorted(SRC_DIR.glob("translator*.spec")):
        source = spec.read_text(encoding="utf-8")
        files = [Path(entry[0]).stem for entry in _spec_list(source, "app_files")]
        modules = _spec_list(source, "app_modules")
        bundled = spec.name not in LITE_WITHOUT_EPUB_LIBRARY
        assert (files.count("epub_library"), modules.count("epub_library")) == ((1, 1) if bundled else (0, 0)), spec.name
    for name in ("library_core",) + U5_SHARED_MODULES:
        tree = ast.parse((SRC_DIR / f"{name}.py").read_text(encoding="utf-8-sig"))
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                top.add(node.module.split(".")[0])
        assert not top & {"epub_library", "PySide6", "translator_gui", "Retranslation_GUI", "dpi_setup"}, (name, top)


def _top_level_imports(name):
    tree = ast.parse((SRC_DIR / f"{name}.py").read_text(encoding="utf-8-sig"))
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            top.add(node.module.split(".")[0])
    return top


def test_u6_modules_are_imported_by_the_desktop_and_stay_gui_free():
    """The U6 cores are module-level imports of their desktop dialogs (hence every spec) and never
    import Qt or the dialogs they were lifted from."""
    for desktop, core in U6_IMPORTERS.items():
        assert core in _top_level_imports(desktop), (desktop, core)
    gui = {"PySide6", "translator_gui", "dpi_setup", "GlossaryManager_GUI", "parallel_epub_glossary",
           "QA_Scanner_GUI", "Retranslation_GUI", "epub_library"}
    for name in U6_SHARED_MODULES:
        assert not _top_level_imports(name) & gui, (name, _top_level_imports(name) & gui)


def test_u7_modules_are_imported_by_the_desktop_and_stay_gui_free():
    """The U7 cores are module-level imports of their desktop / shared callers (hence every spec,
    including the Lite tiers) and never import Qt, the dialogs they were lifted from or the
    manga stack."""
    for importer, cores in U7_IMPORTERS.items():
        for core in cores:
            assert core in _top_level_imports(importer), (importer, core)
    gui = {"PySide6", "translator_gui", "dpi_setup", "Retranslation_GUI", "async_api_processor", "other_settings",
           "review_dialog", "QA_Scanner_GUI", "epub_library", "manga_integration", "manga_translator",
           "ImageRenderer", "bubble_detector", "local_inpainter", "ocr_manager"}
    for name in U7_SHARED_MODULES:
        assert not _top_level_imports(name) & gui, (name, _top_level_imports(name) & gui)
