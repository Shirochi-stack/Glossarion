"""Host tests for the mobile packaging tools (mainly tools/collect_backend.py).

All offline. Most tests build a tiny fake ``src/`` tree. The ``real_repo`` tests run the
collector on the actual repository with the committed backend_manifest.toml; they use a
copy of build/collector_cache.json when one exists, so they stay fast after a collect.

Run: python -m pytest -p no:cacheprovider -W ignore tests_host/test_collect_backend.py
"""
from __future__ import annotations

import ast
import importlib
import json
import shutil
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

MOBILE = Path(__file__).resolve().parents[1]
TOOLS = MOBILE / "tools"
SRC = MOBILE.parent
sys.path.insert(0, str(TOOLS))

cb = importlib.import_module("collect_backend")
vi = importlib.import_module("version_info")
wheels = importlib.import_module("check_mobile_wheels")
assets = importlib.import_module("prepare_assets")

BASE_MANIFEST = """
[entry]
modules = {entries}
planned = ["not_yet_written"]
scan_package = "app/glossarion_mobile"

[exclude]
excluded_mod = "desktop binary"

[gui]
packages = ["PySide6", "tkinter"]
roots = ["gui_root"]

[thirdparty.map]
PIL = "pillow"
yaml = "pyyaml"
"google.genai" = "google-genai"

[thirdparty.unavailable]
torch = "no mobile wheel"

[platform.unavailable]
winreg = "Windows only"

[spawn.wrappers]
"helpers.run_hidden" = "subprocess"

{extra}
"""

BASE_PYPROJECT = """
[project]
name = "x"
version = "0"
dependencies = ["pillow==12.2.0", "google-genai>=1.73,<2", "requests==2.32.5"]
[tool.flet.android]
dependencies = ["psutil==7.2.2"]
"""


def make_tree(tmp_path: Path, files: dict, entries=("entry",), extra: str = "", app_files: dict | None = None,
              pyproject: str = BASE_PYPROJECT):
    src = tmp_path / "src"
    mobile = src / "mobile"
    (mobile / "app" / "glossarion_mobile").mkdir(parents=True)
    for name, body in files.items():
        data = textwrap.dedent(body)
        raw = data.encode("utf-8")
        if name.startswith("BOM:"):
            name, raw = name[4:], b"\xef\xbb\xbf" + raw
        (src / f"{name}.py").write_bytes(raw)
    for rel, body in (app_files or {}).items():
        p = mobile / "app" / "glossarion_mobile" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body), encoding="utf-8")
    manifest = mobile / "backend_manifest.toml"
    manifest.write_text(BASE_MANIFEST.format(entries=json.dumps(list(entries)), extra=textwrap.dedent(extra)),
                        encoding="utf-8")
    pp = mobile / "pyproject.toml"
    pp.write_text(pyproject, encoding="utf-8")
    return src, mobile, manifest, pp


def collect(tmp_path, files, **kw):
    src, mobile, manifest, pp = make_tree(tmp_path, files, **kw)
    m = cb.load_manifest(manifest)
    return cb.Collector(src, m, cb.load_pins(pp), mobile_dir=mobile, cache_path=None).run()


def codes(res, level="error"):
    return sorted({f.code for f in res.findings if f.level == level})


def find(res, code, module=None):
    return [f for f in res.findings if f.code == code and (module is None or f.module == module)]


# ============================================================================ source reading
def test_bom_and_cookie_sources_parse(tmp_path):
    body = b"\xef\xbb\xbf# -*- coding: utf-8 -*-\nimport os\nX = '\xec\x95\x88\xeb\x85\x95'\n"
    with pytest.raises(SyntaxError):  # what a naive reader does with translator_gui.py
        ast.parse(body.decode("utf-8"))
    mi = cb.analyze_source("m", "m.py", body, frozenset())
    assert mi.bom and not mi.error and [i.target for i in mi.imports] == ["os"]
    latin = b"# -*- coding: latin-1 -*-\nS = '\xe9'\nimport json\n"
    mi = cb.analyze_source("l", "l.py", latin, frozenset())
    assert not mi.error and mi.imports[0].target == "json"


def test_real_translator_gui_bom_is_handled():
    path = SRC / "translator_gui.py"
    if not path.exists():
        pytest.skip("desktop translator_gui.py not present")
    data = path.read_bytes()
    assert data.startswith(b"\xef\xbb\xbf"), "translator_gui.py is expected to carry a UTF-8 BOM"
    text = cb.decode_source(data)
    assert not text.startswith("﻿")
    compile(text[:2000].rsplit("\n", 1)[0] + "\n", "probe", "exec", flags=ast.PyCF_ONLY_AST)


# ============================================================================ classification
def test_import_kind_and_guard_classification():
    src = textwrap.dedent('''
        import os
        from typing import TYPE_CHECKING
        from contextlib import suppress
        try:
            import guarded_a
        except ImportError:
            import handler_b
        else:
            import else_c
        try:
            import unguarded_d
        except ValueError:
            pass
        if TYPE_CHECKING:
            import tc_e
        if os.name == "nt":
            import cond_f
        with suppress(ModuleNotFoundError):
            import supp_g
        class K:
            import classbody_h
            def meth(self):
                import lazy_i
        def outer():
            try:
                def inner():
                    import lazy_unguarded_j
            except ImportError:
                pass
            try:
                import lazy_guarded_k
            except Exception:
                pass
        if __name__ == "__main__":
            import main_l
        f = lambda: __import__("lazy_m")
    ''').encode()
    mi = cb.analyze_source("m", "m.py", src, frozenset())
    got = {i.target: (i.kind, i.guarded) for i in mi.imports}
    assert got["guarded_a"] == ("module", True)
    assert got["handler_b"] == ("conditional", False)
    assert got["else_c"] == ("conditional", False)
    assert got["unguarded_d"] == ("module", False)
    assert got["tc_e"] == ("type_checking", False)
    assert got["cond_f"] == ("conditional", False)
    assert got["supp_g"] == ("module", True)
    assert got["classbody_h"] == ("module", False)
    assert got["lazy_i"] == ("lazy", False)
    assert got["lazy_unguarded_j"] == ("lazy", False), "a try around a def does not guard the function body"
    assert got["lazy_guarded_k"] == ("lazy", True)
    assert got["main_l"] == ("main", False)
    assert got["lazy_m"] == ("lazy", False)
    funcs = {i.target: i.func for i in mi.imports}
    assert funcs["lazy_i"] == "K.meth" and funcs["lazy_unguarded_j"] == "outer.<locals>.inner"


def test_namespace_package_targets_keep_submodule():
    mi = cb.analyze_source("m", "m.py", b"from google import genai\nfrom google.auth import x\n"
                                        b"from bs4 import BeautifulSoup, Tag\n", frozenset())
    assert [i.target for i in mi.imports] == ["google.genai", "google.auth.x", "bs4"], "same-line names dedupe"


# ============================================================================ GUI taint / edges
GUI_FILES = {
    "gui_root": "VALUE = 1\n",
    "qt_dialog": "from PySide6.QtWidgets import QDialog\n",
    "uses_dialog": "import qt_dialog\n",                       # tainted transitively
    "guards_dialog": "try:\n    import qt_dialog\nexcept ImportError:\n    qt_dialog = None\n",
    "lazy_dialog": "def show():\n    import qt_dialog\n    return qt_dialog\n",
    "helper": "X = 2\n",
}


def test_gui_taint_propagates_through_module_scope_imports(tmp_path):
    src, mobile, manifest, pp = make_tree(tmp_path, dict(GUI_FILES, entry="import helper\n"))
    c = cb.Collector(src, cb.load_manifest(manifest), cb.load_pins(pp), mobile_dir=mobile)
    assert c.taint("qt_dialog") and "PySide6.QtWidgets" in c.taint("qt_dialog")
    assert "qt_dialog" in c.taint("uses_dialog")
    assert c.taint("gui_root") == "gui_root is a GUI root"
    assert c.taint("guards_dialog") is None
    assert c.taint("lazy_dialog") is None
    assert c.taint("helper") is None


def test_taint_handles_import_cycles(tmp_path):
    files = {"a": "import b\n", "b": "import a\nimport c\n", "c": "import tkinter\n", "entry": "import d\n",
             "d": "import e\n", "e": "import d\n"}
    src, mobile, manifest, pp = make_tree(tmp_path, files)
    c = cb.Collector(src, cb.load_manifest(manifest), cb.load_pins(pp), mobile_dir=mobile)
    assert c.taint("a") and c.taint("b") and c.taint("c")
    assert c.taint("d") is None and c.taint("e") is None


def test_edges_into_gui_code(tmp_path):
    files = dict(GUI_FILES, entry="import guards_dialog\nimport lazy_dialog\nimport helper\n")
    res = collect(tmp_path, files)
    assert {"entry", "guards_dialog", "lazy_dialog", "helper"} == set(res.closure)
    assert res.edge_status["guards_dialog->qt_dialog"] == "guarded"
    assert res.edge_status["lazy_dialog->qt_dialog"] == "FAIL (not baselined)"
    assert codes(res) == ["unbaselined-edge"]

    extra = '[baseline.lazy_gui_edges]\n"lazy_dialog->qt_dialog" = { count = 1, reason = "dialog" }\n' \
            '"helper->gui_root" = { count = 1, reason = "gone" }\n'
    res = collect(tmp_path / "b", files, extra=extra)
    assert res.edge_status["lazy_dialog->qt_dialog"] == "baselined"
    assert not res.errors
    assert [f.message for f in find(res, "stale-baseline")] and find(res, "stale-baseline")[0].level == "warning"


def test_module_scope_import_of_tainted_or_excluded_fails(tmp_path):
    files = dict(GUI_FILES, excluded_mod="X = 1\n",
                 entry="import uses_dialog\n")
    res = collect(tmp_path, files)
    # A module-scope import of tainted code taints the importer itself: the entry is refused.
    assert codes(res) == ["entry-tainted"] and res.closure == []
    files1 = dict(files, entry="import middle\n", middle="def f():\n    pass\nimport uses_dialog\n")
    res = collect(tmp_path / "m", files1)
    assert codes(res) == ["entry-tainted"] and "middle" in find(res, "entry-tainted")[0].message
    files2 = dict(files, entry="import excluded_mod\n")
    res = collect(tmp_path / "x", files2)
    assert res.edge_status["entry->excluded_mod"] == "FAIL (module-scope)"
    assert "excluded_mod" not in res.closure
    files3 = dict(files, entry="try:\n    import excluded_mod\nexcept ImportError:\n    pass\n")
    res = collect(tmp_path / "y", files3)
    assert res.edge_status["entry->excluded_mod"] == "guarded" and not res.errors


def test_edge_ratchet_counts_sites(tmp_path):
    entry = "def a():\n    import excluded_mod\ndef b():\n    import excluded_mod\n"
    extra = '[baseline.lazy_excluded_edges]\n"entry->excluded_mod" = { count = 1, reason = "x" }\n'
    res = collect(tmp_path, {"excluded_mod": "", "entry": entry}, extra=extra)
    assert res.edge_status["entry->excluded_mod"].startswith("FAIL (count 2 > 1)")
    res = collect(tmp_path / "ok", {"excluded_mod": "", "entry": entry}, extra=extra.replace("count = 1", "count = 3"))
    assert not res.errors and find(res, "edge-ratchet-slack")


def test_main_block_and_gui_package_edges_need_baseline(tmp_path):
    entry = ("def show():\n    from PySide6.QtWidgets import QMessageBox\n"
             "if __name__ == '__main__':\n    import gui_root\n")
    res = collect(tmp_path, {"gui_root": "", "entry": entry})
    assert res.edge_status["entry->PySide6"] == "FAIL (not baselined)"
    assert res.edge_status["entry->gui_root"] == "FAIL (not baselined)"
    res = collect(tmp_path / "q", {"entry": "import PySide6\n"})
    assert find(res, "entry-tainted"), "a module-scope Qt import taints the entry itself"


def test_entry_errors(tmp_path):
    res = collect(tmp_path, {"entry": "", "excluded_mod": ""}, entries=("entry", "missing", "excluded_mod"))
    assert {"entry-missing", "entry-excluded"} <= set(codes(res))
    res = collect(tmp_path / "p", {"entry": "", "not_yet_written": ""})
    assert find(res, "planned-present") and "not_yet_written" in res.closure


# ============================================================================ process-spawn ratchet
SPAWN_MODULE = '''
import os
import subprocess
import subprocess as _sp
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor as PPE, ThreadPoolExecutor
import concurrent.futures
from helpers import run_hidden

_RUN = subprocess.run

def compress():
    p = _sp.Popen(["x"])
    try:
        subprocess.check_call(["y"])
    except subprocess.CalledProcessError:
        pass

def pools(flag):
    cls = PPE if flag else ThreadPoolExecutor
    with cls(2) as ex:
        pass
    ctx = mp.get_context("spawn")
    mp.get_context("fork").Pool(2)
    q = mp.Queue()
    return concurrent.futures.ProcessPoolExecutor(max_workers=1)

def aliased():
    _RUN(["z"])
    os.system("ls")
    run_hidden(["w"])

def safe():
    def run(x):
        return x
    run(1)
    isinstance(None, PPE)
    ThreadPoolExecutor(2)

class Worker(mp.Process):
    pass
'''


def test_spawn_detection_resolves_aliases():
    mi = cb.analyze_source("s", "s.py", SPAWN_MODULE.encode(), frozenset({"helpers"}),
                           spawn_apis={**cb.SPAWN_APIS, "helpers.run_hidden": "subprocess"})
    sites = {(s.func, s.api, s.how) for s in mi.spawns}
    assert ("compress", "subprocess.Popen", "call") in sites, "import subprocess as _sp; _sp.Popen"
    assert ("compress", "subprocess.check_call", "call") in sites
    assert ("pools", "concurrent.futures.ProcessPoolExecutor", "reference") in sites, "cls = PPE if ... else ..."
    assert ("pools", "multiprocessing.get_context", "call") in sites
    assert ("pools", "multiprocessing.Pool", "call") in sites, "mp.get_context(...).Pool"
    assert ("pools", "multiprocessing.Queue", "call") in sites
    assert ("pools", "concurrent.futures.ProcessPoolExecutor", "call") in sites
    assert ("aliased", "subprocess.run", "call") in sites, "module-level _RUN = subprocess.run"
    assert ("aliased", "os.system", "call") in sites
    assert ("aliased", "helpers.run_hidden", "call") in sites, "[spawn].wrappers"
    assert ("<module>", "multiprocessing.Process", "reference") in sites, "class Worker(mp.Process)"
    assert not any(f == "safe" for f, _, _ in sites), "shadowed names, isinstance and threads are not spawns"
    assert not any("CalledProcessError" in a for _, a, _ in sites)


def test_spawn_ratchet(tmp_path):
    files = {"helpers": "", "entry": "import subprocess as _sp\ndef go():\n    _sp.Popen(['x'])\n"}
    res = collect(tmp_path, files)
    assert codes(res) == ["spawn-new"] and res.spawn_sites["entry:go"]["status"] == "FAIL (new)"

    extra = ('[baseline.process_spawn_sites]\n'
             '"entry:go" = { apis = ["subprocess.Popen"], count = 1, reason = "gated in U1" }\n')
    res = collect(tmp_path / "ok", files, extra=extra)
    assert not res.errors and res.spawn_sites["entry:go"]["status"] == "baselined"

    more = dict(files, entry=files["entry"] + "    _sp.Popen(['y'])\n")
    res = collect(tmp_path / "count", more, extra=extra)
    assert codes(res) == ["spawn-ratchet"]

    newapi = dict(files, entry=files["entry"] + "    _sp.run(['y'])\n")
    res = collect(tmp_path / "api", newapi, extra=extra.replace("count = 1", "count = 2"))
    assert codes(res) == ["spawn-new-api"]

    other = dict(files, entry=files["entry"] + "def other():\n    import os\n    os.fork()\n")
    res = collect(tmp_path / "func", other, extra=extra)
    assert codes(res) == ["spawn-new"] and "entry:other" in res.spawn_sites

    gone = dict(files, entry="def go():\n    pass\n")
    res = collect(tmp_path / "stale", gone, extra=extra)
    assert not res.errors and find(res, "stale-baseline")


# ============================================================================ third-party, platform, dynamic, file refs
def test_thirdparty_mapping_and_pins(tmp_path):
    entry = textwrap.dedent('''
        from google import genai
        import PIL.Image
        import requests
        def later():
            import yaml
            import torch
        try:
            import psutil
        except ImportError:
            psutil = None
    ''')
    res = collect(tmp_path, {"entry": entry})
    assert not res.errors
    tp = res.thirdparty
    assert tp["google-genai"]["status"] == "pinned" and tp["pillow"]["status"] == "pinned"
    assert tp["pyyaml"]["status"] == "unpinned" and tp["torch"]["status"] == "unavailable"
    assert tp["psutil"]["status"] == "android-only"

    bad = collect(tmp_path / "bad", {"entry": "import yaml\nimport torch\nimport psutil\nimport winreg\nimport imp\n"})
    assert set(codes(bad)) == {"unpinned-import", "unavailable-import", "android-only-import", "platform-import",
                               "removed-stdlib"}
    cond = collect(tmp_path / "cond", {"entry": "import sys\nif sys.platform == 'win32':\n    import winreg\n"})
    assert not cond.errors and cond.thirdparty["stdlib:winreg"]["status"] == "platform-unavailable"


def test_dynamic_imports(tmp_path):
    files = {"plug_a": "", "plug_b": "", "lit": "",
             "entry": "import importlib\nimportlib.import_module('lit')\n"
                      "def load(name):\n    return __import__(name)\n"}
    res = collect(tmp_path, files)
    assert "lit" in res.closure and codes(res) == ["dynamic-import"]
    extra = '[dynamic_imports]\n"entry:load" = ["plug_a", "plug_b"]\n'
    res = collect(tmp_path / "ok", files, extra=extra)
    assert not res.errors and {"plug_a", "plug_b", "lit"} <= set(res.closure)


def test_gui_file_reference(tmp_path):
    files = {"gui_root": "", "worker": "",
             "entry": "import os, sys\nW = os.path.join(os.path.dirname(__file__), 'worker.py')\n"
                      "def go():\n    return open(os.path.join('x', 'gui_root.py'))\n"}
    res = collect(tmp_path, files)
    assert codes(res) == ["gui-file-ref"]
    assert find(res, "script-ref-outside-bundle"), "worker.py is referenced but not bundled"
    res = collect(tmp_path / "ok", files, extra='[baseline.gui_file_refs]\n"entry->gui_root.py" = "reads text only"\n')
    assert not res.errors


def test_app_package_scan(tmp_path):
    app = {"__init__.py": "", "jobs.py": "def run():\n    import helper\n    import flet\n    import main\n",
           "ui.py": "import tkinter\n"}
    res = collect(tmp_path, {"entry": "", "helper": "", "main": "import PySide6\n"}, app_files=app,
                  pyproject=BASE_PYPROJECT.replace('"requests==2.32.5"', '"requests==2.32.5", "flet==1.0.3"'))
    assert "helper" in res.closure and res.reasons["helper"].startswith("app ")
    assert "main" not in res.closure, "app-local `main` (app/main.py) is not src/main.py"
    assert codes(res) == ["app-gui-import"]


def test_compile_errors_are_reported(tmp_path):
    res = collect(tmp_path, {"entry": "def f():\n    nonlocal x\n"})
    assert codes(res) == ["compile"]


# ============================================================================ bundle output / verify / CLI
def test_cli_writes_bundle_and_verify_detects_tampering(tmp_path, capsys):
    files = {"entry": "import helper\n", "helper": "X = 1\n", "app_version": 'APP_VERSION = "1.2.3"\n'}
    src, mobile, manifest, pp = make_tree(tmp_path, files, entries=("entry", "app_version"))
    out = tmp_path / "bundle"
    args = ["--src", str(src), "--manifest", str(manifest), "--pyproject", str(pp), "--out", str(out),
            "--report", str(tmp_path / "r.md"), "--json", str(tmp_path / "r.json"), "--no-cache"]
    assert cb.main(args + ["--check"]) == 0 and not out.exists(), "--check writes no bundle"
    assert cb.main(args) == 0
    assert sorted(p.name for p in out.iterdir()) == ["_bundle_info.py", "app_version.py", "entry.py", "helper.py"]
    info = cb.load_bundle_info(out)
    assert info["BUILD_VERSION"] == "1.2.3" and info["MODULES"] == ("app_version", "entry", "helper")
    assert set(info["FILES"]) == {"app_version.py", "entry.py", "helper.py"}
    assert info["BUNDLE_SHA256"] == cb.bundle_digest(info["FILES"])
    report = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "Result: **PASS**" in report and "`helper`" in report
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["closure"] == ["app_version", "entry", "helper"]

    verify = ["--src", str(src), "--out", str(out), "--verify"]
    assert cb.main(verify) == 0
    (out / "helper.py").write_text("X = 2\n", encoding="utf-8")
    assert cb.main(verify) == 1 and "hash mismatch: helper.py" in capsys.readouterr().err
    (out / "helper.py").write_bytes((src / "helper.py").read_bytes())
    (out / "extra.py").write_text("", encoding="utf-8")
    assert cb.main(verify) == 1
    (out / "extra.py").unlink()
    (src / "helper.py").write_text("X = 3\n", encoding="utf-8")
    assert cb.main(verify) == 1 and "stale bundle: helper.py" in capsys.readouterr().err
    assert cb.verify_bundle(out, None) == [], "without src only integrity is checked"


def test_cli_refuses_to_wipe_foreign_dir(tmp_path):
    src, mobile, manifest, pp = make_tree(tmp_path, {"entry": ""})
    out = tmp_path / "precious"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        cb.main(["--src", str(src), "--manifest", str(manifest), "--pyproject", str(pp), "--out", str(out),
                 "--no-cache"])
    assert (out / "keep.txt").exists()


def test_policy_failure_exits_1_without_bundle(tmp_path):
    src, mobile, manifest, pp = make_tree(tmp_path, {"entry": "import yaml\n"})
    out = tmp_path / "bundle"
    rc = cb.main(["--src", str(src), "--manifest", str(manifest), "--pyproject", str(pp), "--out", str(out),
                  "--no-cache", "-q"])
    assert rc == 1 and not out.exists()


def test_cache_roundtrip(tmp_path):
    files = {"entry": "import subprocess\ndef go():\n    subprocess.run(['x'])\n"}
    src, mobile, manifest, pp = make_tree(tmp_path, files)
    cache = tmp_path / "cache.json"
    m = cb.load_manifest(manifest)
    r1 = cb.Collector(src, m, cb.load_pins(pp), mobile_dir=mobile, cache_path=cache).run()
    assert cache.exists()
    r2 = cb.Collector(src, m, cb.load_pins(pp), mobile_dir=mobile, cache_path=cache).run()
    assert r1.spawn_sites == r2.spawn_sites and r1.closure == r2.closure


# ============================================================================ the real repository
@pytest.fixture(scope="session")
def repo_cache(tmp_path_factory):
    if not (SRC / "TransateKRtoEN.py").exists():
        pytest.skip("desktop sources not present")
    dst = tmp_path_factory.mktemp("collector") / "cache.json"
    if cb.DEFAULT_CACHE.exists():
        shutil.copyfile(cb.DEFAULT_CACHE, dst)
    return dst


@pytest.fixture(scope="session")
def real_result(repo_cache):
    m = cb.load_manifest(cb.DEFAULT_MANIFEST)
    return cb.Collector(SRC, m, cb.load_pins(cb.DEFAULT_PYPROJECT), mobile_dir=MOBILE, cache_path=repo_cache).run()


def test_real_repo_passes_policy(real_result):
    res = real_result
    assert not res.errors, "\n".join(f"{f.code} {f.module} {f.lines[:5]}: {f.message}" for f in res.errors)
    closure = set(res.closure)
    for must in ("TransateKRtoEN", "unified_api_client", "Chapter_Extractor", "epub_converter", "scan_html_folder",
                 "extract_glossary_from_epub", "GlossaryManager", "chapter_extraction_worker", "shutdown_utils"):
        assert must in closure
    for never in ("translator_gui", "other_settings", "epub_library", "Retranslation_GUI", "QA_Scanner_GUI",
                  "antigravity_proxy", "dpi_setup", "splash_utils"):
        assert never not in closure


def test_real_repo_known_spawn_sites_are_found(real_result):
    sites = real_result.spawn_sites
    assert "subprocess.Popen" in sites["epub_converter:EPUBCompiler._compress_images"]["apis"], "_sp.Popen alias"
    assert "concurrent.futures.ProcessPoolExecutor" in sites["glossary_process_worker:generate_glossary_async"]["apis"]
    assert "subprocess.run" in sites["pdf_extractor:_extract_with_pdf2htmlex"]["apis"]
    assert "concurrent.futures.ProcessPoolExecutor" in sites["Chapter_Extractor:_extract_chapters_universal"]["apis"]


def test_real_repo_ratchet_fails_on_new_violations(tmp_path, repo_cache):
    src = tmp_path / "src"
    src.mkdir()
    for p in SRC.glob("*.py"):
        shutil.copyfile(p, src / p.name)
    # A new spawn through an alias, a new lazy Qt import and a module-scope import of an
    # excluded module, each in a bundled module.
    with open(src / "language_options.py", "a", encoding="utf-8") as f:
        f.write("\n\ndef _new_spawn():\n    import subprocess as _x\n    return _x.Popen(['true'])\n")
    with open(src / "history_manager.py", "a", encoding="utf-8") as f:
        f.write("\n\ndef _new_dialog():\n    from PySide6.QtWidgets import QMessageBox\n    return QMessageBox\n")
    with open(src / "app_version.py", "a", encoding="utf-8") as f:
        f.write("\nimport tor_proxy\n")
    cache = tmp_path / "cache.json"
    if repo_cache.exists():
        shutil.copyfile(repo_cache, cache)
    m = cb.load_manifest(cb.DEFAULT_MANIFEST)
    res = cb.Collector(src, m, cb.load_pins(cb.DEFAULT_PYPROJECT), mobile_dir=MOBILE, cache_path=cache).run()
    got = {(f.code, f.module) for f in res.errors}
    assert ("spawn-new", "language_options") in got
    assert ("unbaselined-edge", "history_manager") in got
    assert ("eager-blocked-edge", "app_version") in got
    assert len(res.errors) == 3, [f"{f.code} {f.module}: {f.message}" for f in res.errors]


def test_manifest_baselines_are_tight(real_result):
    """Every baselined count equals today's count (nothing to tighten, nothing stale)."""
    res = real_result
    assert not [f for f in res.findings if f.code.endswith("slack") or f.code.startswith("stale")]


# ============================================================================ sibling tools (offline parts)
def test_version_info_build_number(tmp_path):
    p = tmp_path / "app_version.py"
    p.write_bytes(b"\xef\xbb\xbfAPP_VERSION = \"9.13.6\"\n")
    info = vi.compute(p, rebuild=0, github_ref="refs/tags/v9.13.6")
    assert info["build_number"] == 9130600 and info["tag"] == "v9.13.6" and info["is_release"]
    assert info["artifact_prefix"] == "Glossarion_v9.13.6" and info["tag_matches"]
    assert vi.build_number("10.0.1", 3) == 10000103
    assert vi.compute(p, rebuild=0, github_ref="refs/heads/main")["is_release"] is False
    with pytest.raises(ValueError):
        vi.build_number("9.100.0")
    if (SRC / "app_version.py").exists():
        real = vi.compute(rebuild=0, github_ref="")
        assert real["build_number"] == vi.build_number(real["build_version"])


def test_wheel_version_specifier_and_markers():
    V = wheels.Version
    assert V("1.0.0") == V("1.0") and V("1.0rc1") < V("1.0") < V("1.0.post1") and V("1.0.dev1") < V("1.0a1")
    s = wheels.SpecifierSet(">=2.32,<3")
    assert s.contains(V("2.54.0")) and not s.contains(V("3.0.0")) and not s.contains(V("3.0.0rc1"))
    assert wheels.SpecifierSet("~=1.4.2").contains(V("1.4.9")) and not wheels.SpecifierSet("~=1.4.2").contains(V("1.5"))
    assert wheels.SpecifierSet("==1.2.*").contains(V("1.2.7")) and not wheels.SpecifierSet("!=1.2.*").contains(V("1.2.0"))
    env = wheels.ANDROID_ENV
    assert wheels.evaluate_marker('platform_system != "Emscripten"', env)
    assert not wheels.evaluate_marker('python_version < "3.11"', env)
    assert wheels.evaluate_marker('sys_platform == "linux" and (platform_system == "Android" or os_name == "nt")', env)
    assert wheels.evaluate_marker('extra == "socks"', env, frozenset({"socks"}))
    assert not wheels.evaluate_marker('extra == "socks"', env)
    req = wheels.parse_requirement('flet-cli==1.0.3; extra == "cli"')
    assert req.key == "flet-cli" and str(req.spec) == "==1.0.3" and req.marker == 'extra == "cli"'


def test_wheel_tag_compatibility():
    t = wheels.TARGETS
    f = wheels.parse_filename("lxml-6.1.1-1-cp313-cp313-android_24_arm64_v8a.whl", "lxml", "flet")
    assert wheels.file_ok(f, t["arm64-v8a"], False) and not wheels.file_ok(f, t["x86_64"], False)
    ios = wheels.parse_filename("lxml-6.1.1-1-cp313-cp313-ios_13_0_arm64_iphonesimulator.whl", "lxml", "flet")
    assert wheels.file_ok(ios, t["iphonesimulator.arm64"], False) and not wheels.file_ok(ios, t["iphoneos.arm64"], False)
    abi3 = wheels.parse_filename("cryptography-43.0.1-cp37-abi3-android_24_x86_64.whl", "cryptography", "flet")
    assert wheels.file_ok(abi3, t["x86_64"], False)
    old = wheels.parse_filename("lxml-6.1.1-1-cp312-cp312-android_24_arm64_v8a.whl", "lxml", "flet")
    assert not wheels.file_ok(old, t["arm64-v8a"], False)
    py2 = wheels.parse_filename("langdetect-1.0.9-py2-none-any.whl", "langdetect", "pypi")
    pure = wheels.parse_filename("six-1.17.0-py2.py3-none-any.whl", "six", "pypi")
    assert not wheels.file_ok(py2, t["arm64-v8a"], False) and wheels.file_ok(pure, t["arm64-v8a"], False)
    sdist = wheels.parse_filename("langdetect-1.0.9.tar.gz", "langdetect", "pypi")
    assert wheels.file_ok(sdist, t["arm64-v8a"], True) and not wheels.file_ok(pure, t["arm64-v8a"], True)
    host = wheels.parse_filename("numpy-2.4.6-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl", "numpy",
                                 "pypi")
    assert wheels.file_ok(host, wheels.HOST_TARGETS[0], False)


class _FakeFetcher:
    """Offline stand-in for check_mobile_wheels.Fetcher."""

    def __init__(self, files: dict, deps: dict, flet: set):
        self._files, self._deps, self._flet = files, deps, flet
        self.requests = 0

    def flet_projects(self):
        return {n: f"{n}/" for n in self._flet}

    def files(self, name):
        return [wheels.parse_filename(fn, name, "flet" if name in self._flet else "pypi")
                for fn in self._files.get(name, [])]

    def requires_dist(self, name, version):
        return self._deps.get((name, str(version)), [])

    def prefetch(self, fn, items):
        for it in items:
            fn(*it) if isinstance(it, tuple) else fn(it)


def test_wheel_resolver_finds_transitive_gaps():
    files = {
        "app": ["app-1.0-py3-none-any.whl"],
        "fastlib": ["fastlib-2.0-cp313-cp313-android_24_arm64_v8a.whl", "fastlib-2.0-cp313-cp313-manylinux_2_17_x86_64.whl"],
        "nowheel": ["nowheel-1.0-cp313-cp313-manylinux_2_17_x86_64.whl", "nowheel-1.0.tar.gz"],
        "grpcio": ["grpcio-1.81.0-cp313-cp313-android_24_arm64_v8a.whl", "grpcio-1.84.0-cp313-cp313-win_amd64.whl"],
        "grpcio-status": ["grpcio_status-1.84.0-py3-none-any.whl"],
    }
    deps = {("app", "1.0"): ["fastlib>=1", "nowheel; platform_system == 'Android'", "skipme; sys_platform == 'win32'"],
            ("grpcio-status", "1.84.0"): ["grpcio>=1.84.0"]}
    fetcher = _FakeFetcher(files, deps, {"fastlib", "grpcio"})
    rv = wheels.Resolver(wheels.TARGETS["arm64-v8a"], fetcher, set(), set())
    rv.resolve([wheels.parse_requirement("app"), wheels.parse_requirement("grpcio==1.81.0"),
                wheels.parse_requirement("grpcio-status==1.84.0")])
    assert str(rv.chosen["fastlib"].version) == "2.0"
    assert "skipme" not in rv.chosen
    errs = "\n".join(rv.errors)
    assert "nowheel" in errs and "needed by app" in errs
    assert "grpcio (==1.81.0, >=1.84.0" in errs
    rv2 = wheels.Resolver(wheels.TARGETS["arm64-v8a"], fetcher, {"nowheel"}, set())
    rv2.resolve([wheels.parse_requirement("app")])
    assert rv2.chosen["nowheel"].file.kind == "sdist" and not rv2.errors


def test_prepare_assets_selftest_epub_is_reproducible(tmp_path):
    pytest.importorskip("ebooklib")
    a = assets.prepare_selftest(tmp_path / "a", log=lambda *_: None)
    b = assets.prepare_selftest(tmp_path / "b", log=lambda *_: None)
    assert a["sha256"] == b["sha256"] and a["chapters"] == 12
    epub_path = tmp_path / "a" / assets.SELFTEST_EPUB
    with zipfile.ZipFile(epub_path) as zf:
        names = zf.namelist()
        assert names[0] == "mimetype" and zf.getinfo("mimetype").compress_type == zipfile.ZIP_STORED
        chapters = [n for n in names if n.startswith("EPUB/text/chapter")]
        assert len(chapters) == 12, "more than 10 chapters so Chapter_Extractor uses its pool path"
        text = "".join(zf.read(n).decode("utf-8") for n in chapters)
    for name in assets.CHARACTERS + assets.TERMS:
        assert name in text
    import tomllib
    manifest = tomllib.loads((tmp_path / "a" / "MANIFEST.toml").read_text(encoding="utf-8"))
    assert manifest["epub"]["sha256"] == a["sha256"] and manifest["epub"]["chapters"] == 12
    assert assets.prepare_selftest(tmp_path / "a", check=True, log=lambda *_: None)["sha256"] == a["sha256"]


def test_prepare_assets_tiktoken_from_local_cache(tmp_path, monkeypatch):
    fake = {"cl100k_base": ("https://example.invalid/cl100k_base.tiktoken", None),
            "o200k_base": ("https://example.invalid/o200k_base.tiktoken", None)}
    cache = tmp_path / "cache"
    cache.mkdir()
    table = {}
    for name, (url, _) in fake.items():
        data = f"{name} 0\n".encode()
        (cache / assets.cache_key(url)).write_bytes(data)
        table[name] = (url, assets.sha256(data))
    monkeypatch.setattr(assets, "TIKTOKEN_ENCODINGS", table)
    monkeypatch.setattr(assets, "installed_tiktoken_constants", lambda: {})
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache))
    out = tmp_path / "tiktoken"
    entries = assets.prepare_tiktoken(out, offline=True, log=lambda *_: None)
    import hashlib
    for name, (url, digest) in table.items():
        key = hashlib.sha1(url.encode()).hexdigest()
        assert entries[name]["cache_key"] == key and (out / key).exists()
    assert assets.prepare_tiktoken(out, check=True, log=lambda *_: None)
    (out / entries["cl100k_base"]["cache_key"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError):
        assets.prepare_tiktoken(out, check=True, log=lambda *_: None)
