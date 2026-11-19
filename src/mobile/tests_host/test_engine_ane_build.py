"""The on-device engine's optional NPU backend on iOS in the build (wheelhouse.toml [ane], recipes/llama-cpp-python:
meta.yaml, patches/ane-backend.patch and mobile.patch 11., ci/wheels/wheels.py, ci/verify_ipa.py): the manifest
switch, stage-ane's CMake initial cache, the recipe and the patch (the pull request's backend with the Glossarion
changes its header lists), the verify rules for libggml-ane and its marker (fake Mach-O files), the simulator
self-test assertion, the IPA check and the patched loader's iOS functions over a fake ctypes. No network, no Mac,
no device.
"""

from __future__ import annotations

import collections
import importlib.util
import io
import os
import platform
import re
import types
import zipfile
from pathlib import Path

import pytest

import test_check_mobile_wheels_wheelhouse as base
from test_check_mobile_wheels_wheelhouse import (ARM64, IOS_LLAMA_LIBS, IOS_LLAMA_NEEDED, IOS_SLICES, METAL_FRAMEWORKS,
                                                 MOBILE, _FakeFn, _llama_dist_info, _patched_cpu_loader, _render_recipe,
                                                 _verify_llama, _zip_tree, make_macho, wh)
from test_engine_npu_build import (ANE_PATCH, COMMIT, PATCH, RECIPE, RELEASE, _device_libggml, _file_diffs,
                                   _real_ctypes_ns, _steps)

PR, PR_COMMIT = 31969, "a45d7d2e"
ANE_STEPS = ("shipped", "device", "framework", "registry", "compile")
ANE_ENV = ("GGML_ANE_WEIGHTS", "GGML_ANE_SEG_WEIGHT_MB", "GGML_ANE_PURGE_CACHE", "GGML_ANE_DUMP_DIR",
           "GGML_ANE_PROFILE", "GLOSSARION_GGML_EAGER_ANE")
EXPORTS = ("ggml_backend_ane_reg", "ggml_backend_ane_collect_shapes", "ggml_backend_ane_set_kv_buckets",
           "glossarion_ane_probe", "glossarion_ane_selftest", "glossarion_ane_freeze", "glossarion_ane_set_cpu_threads",
           "glossarion_ane_get_stats", "glossarion_ane_reset_stats")


# ---- the manifest switch --------------------------------------------------------------------------------------------

def _manifest(tmp_path: Path, enabled: bool = True, table: str | None = None) -> Path:
    """A copy of the real wheelhouse.toml (absolute recipe paths) with [ane] switched on or off (or replaced)."""
    text = wh.DEFAULT_MANIFEST.read_text(encoding="utf-8").replace("\r\n", "\n")
    recipes = (MOBILE / "ci" / "wheels" / "recipes").as_posix()
    text = text.replace('recipe = "recipes/', f'recipe = "{recipes}/')
    old = re.search(r"^\[ane\]\n(?:[^\[\n].*\n)*", text, re.M).group(0)
    assert old.startswith("[ane]\nenabled = true\n")
    text = text.replace(old, table if table is not None else old.replace("enabled = true", f"enabled = {str(enabled).lower()}"))
    path = tmp_path / "manifest" / "wheelhouse.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_manifest_switches_the_backend_on_for_the_three_ios_slices():
    ios = wh.Ctx(wh.DEFAULT_MANIFEST, "ios")
    assert wh.ane_spec(ios.data) == {"targets": wh.ANE_TARGETS, "pr": PR, "commit": PR_COMMIT}, \
        "[ane] stays enabled on this branch"
    assert wh.ANE_TARGETS == ("iphoneos.arm64", "iphonesimulator.arm64", "iphonesimulator.x86_64")
    assert wh.ane_targets(ios) == wh.ANE_TARGETS
    assert wh.builds_ane_recipe(ios, ["llama-cpp-python"]) and not wh.builds_ane_recipe(ios, ["cryptography"])
    assert not wh.builds_ane_recipe(wh.Ctx(wh.DEFAULT_MANIFEST, "android"), ["llama-cpp-python"])
    # Android's NPU switch is its own table
    assert wh.hexagon_targets(ios) and "ane" not in wh.hexagon_spec(ios.data)


def test_switched_off_manifest_expects_nothing(tmp_path):
    ctx = wh.Ctx(_manifest(tmp_path, enabled=False), "ios", wh.DEFAULT_PYPROJECT)
    assert wh.ane_spec(ctx.data) is None and wh.ane_targets(ctx) == ()
    assert wh.builds_ane_recipe(ctx, ["llama-cpp-python"])                 # the recipe still loads the cache (OFF)
    profile = wh.SITE_PROFILES["llama-cpp-python"]
    for target in wh.ANE_TARGETS:
        assert wh.profile_libs(profile, target, {}, wh.ane_targets(ctx)) == IOS_LLAMA_LIBS
        assert wh.profile_libs(profile, target, {}, wh.ANE_TARGETS) == IOS_LLAMA_LIBS + ("libggml-ane",)
    # an Android target never gets it
    assert "libggml-ane" not in wh.profile_libs(profile, "arm64-v8a", {}, wh.ANE_TARGETS)


@pytest.mark.parametrize("table, message", [
    ('[ane]\nenabled = "yes"\n', "[ane] needs enabled = true or false"),
    ('[ane]\nenabled = true\ntargets = ["iphoneos.arm64"]\npr = 1\ncommit = "abcdef0"\n', "[ane] targets must be"),
    ('[ane]\nenabled = true\ntargets = ["iphonesimulator.arm64", "iphoneos.arm64", "iphonesimulator.x86_64"]\n'
     'pr = 1\ncommit = "abcdef0"\n', "[ane] targets must be"),
    ('[ane]\nenabled = true\ntargets = ["iphoneos.arm64", "iphonesimulator.arm64", "iphonesimulator.x86_64"]\n'
     'pr = 0\ncommit = "abcdef0"\n', "[ane] needs pr"),
    ('[ane]\nenabled = true\ntargets = ["iphoneos.arm64", "iphonesimulator.arm64", "iphonesimulator.x86_64"]\n'
     'pr = 31969\ncommit = "main"\n', "[ane] needs pr"),
])
def test_bad_ane_tables_are_refused(table, message):
    with pytest.raises(wh.WheelsError, match=re.escape(message)):
        wh.ane_spec(wh.cmw.tomllib.loads(table))
    assert wh.ane_spec({}) is None and wh.ane_spec({"ane": {"enabled": False}}) is None


def test_env_exports_the_switch_on_ios_only(tmp_path, capsys):
    args = types.SimpleNamespace(format="github", constraints_out=None)
    wh.cmd_env(wh.Ctx(wh.DEFAULT_MANIFEST, "ios"), args)
    assert "ane_enabled=true" in capsys.readouterr().out.splitlines()
    wh.cmd_env(wh.Ctx(_manifest(tmp_path, enabled=False), "ios", wh.DEFAULT_PYPROJECT), args)
    assert "ane_enabled=false" in capsys.readouterr().out.splitlines()
    wh.cmd_env(wh.Ctx(wh.DEFAULT_MANIFEST, "android"), args)
    assert "ane_enabled" not in capsys.readouterr().out


# ---- stage-ane ------------------------------------------------------------------------------------------------------

def _xcframework(tmp_path: Path, slices=wh.ANE_XCFRAMEWORK_SLICES) -> Path:
    xcf = tmp_path / "support" / "3.13" / "iOS" / "Python.xcframework"
    for s in slices:
        (xcf / s / "include").mkdir(parents=True)
    return xcf


def test_stage_ane_writes_the_initial_cache_into_both_slices(tmp_path, capsys):
    xcf = _xcframework(tmp_path)
    argv = ["--manifest", str(_manifest(tmp_path)), "stage-ane", "--platform", "ios", "--xcframework", str(xcf),
            "llama-cpp-python"]
    assert wh.main(argv) == 0
    for s in ("ios-arm64", "ios-arm64_x86_64-simulator"):
        lines = (xcf / s / "glossarion" / "ane-init.cmake").read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("# Written by src/mobile/ci/wheels/wheels.py stage-ane")
        assert lines[1:] == ['set(GGML_ANE ON CACHE BOOL "" FORCE)']
    # switched off: OFF in both
    argv[1] = str(_manifest(tmp_path / "off", enabled=False))
    assert wh.main(argv) == 0
    for s in wh.ANE_XCFRAMEWORK_SLICES:
        assert (xcf / s / wh.ANE_INIT_CACHE).read_text(encoding="utf-8").splitlines()[1:] == \
            ['set(GGML_ANE OFF CACHE BOOL "" FORCE)']
    assert wh.ANE_INIT_CACHE == "glossarion/ane-init.cmake"


def test_stage_ane_without_the_engine_or_the_slices(tmp_path, capsys):
    xcf = _xcframework(tmp_path, slices=("ios-arm64",))
    manifest = str(_manifest(tmp_path))
    # a run without llama-cpp-python (or an Android run) writes nothing
    assert wh.main(["--manifest", manifest, "stage-ane", "--platform", "ios", "--xcframework", str(xcf),
                    "cryptography"]) == 0
    assert wh.main(["--manifest", manifest, "stage-ane", "--platform", "android", "--xcframework", str(xcf)]) == 0
    assert not (xcf / "ios-arm64" / "glossarion").exists()
    # a missing slice is an error before anything is written
    assert wh.main(["--manifest", manifest, "stage-ane", "--platform", "ios", "--xcframework", str(xcf),
                    "llama-cpp-python"]) == 1
    assert "no slice ios-arm64_x86_64-simulator" in capsys.readouterr().err
    assert not (xcf / "ios-arm64" / "glossarion").exists()


#: the parts of mobile-forge (pinned commit 0fadbfab) the recipe's -C{HOST_PYTHON_HOME}/... relies on, as they read
_FORGE_BUILD = '''
        script_vars = {
            "HOST_PYTHON_HOME": str(self.cross_venv.host_python_home),
        }
        for key, value in self.package.meta["build"]["script_env"].items():
            env[key] = str(value).format(**script_vars)
'''
_FORGE_CROSS = '''
class CrossVEnv:
    XCFRAMEWORK_SLICES = {
        ("iphonesimulator", "arm64"): "ios-arm64_x86_64-simulator",
        ("iphonesimulator", "x86_64"): "ios-arm64_x86_64-simulator",
        ("iphoneos", "arm64"): "ios-arm64",
        ("appletvos", "arm64"): "tvos-arm64",
    }
'''


def _forge(tmp_path: Path, build: str = _FORGE_BUILD, cross: str = _FORGE_CROSS) -> Path:
    src = tmp_path / "forge" / "src" / "forge"
    src.mkdir(parents=True)
    (src / "build.py").write_text(build, encoding="utf-8")
    (src / "cross.py").write_text(cross, encoding="utf-8")
    return tmp_path / "forge"


def test_stage_ane_checks_that_forge_still_points_host_python_home_at_the_slices(tmp_path, capsys):
    assert wh.forge_ane_errors(_forge(tmp_path / "ok")) == []
    xcf = _xcframework(tmp_path)
    manifest = str(_manifest(tmp_path))
    argv = ["--manifest", manifest, "stage-ane", "--platform", "ios", "--xcframework", str(xcf), "--forge",
            str(_forge(tmp_path / "ok2")), "llama-cpp-python"]
    assert wh.main(argv) == 0 and (xcf / "ios-arm64" / wh.ANE_INIT_CACHE).is_file()
    # a forge bump that drops the key, renames a slice or stops formatting script_env fails before anything is written
    for name, build, cross, needle in (
            ("key", _FORGE_BUILD.replace("HOST_PYTHON_HOME", "PYTHON_HOME"), _FORGE_CROSS, "HOST_PYTHON_HOME key"),
            ("format", _FORGE_BUILD.replace(".format(**script_vars)", ""), _FORGE_CROSS, "HOST_PYTHON_HOME key"),
            ("slice", _FORGE_BUILD, _FORGE_CROSS.replace('"ios-arm64"', '"ios-arm64e"'), "XCFRAMEWORK_SLICES"),
            ("gone", _FORGE_BUILD, "class CrossVEnv:\n    pass\n", "XCFRAMEWORK_SLICES")):
        forge = _forge(tmp_path / name, build, cross)
        assert any(needle in e for e in wh.forge_ane_errors(forge)), name
        fresh = _xcframework(tmp_path / f"x-{name}")
        argv[6], argv[8] = str(fresh), str(forge)
        assert wh.main(argv) == 1, name
        assert needle in capsys.readouterr().err
        assert not (fresh / "ios-arm64" / "glossarion").exists()
    assert wh.forge_ane_errors(tmp_path / "nowhere")[0].endswith(")") and "cannot read" in \
        wh.forge_ane_errors(tmp_path / "nowhere")[0]


def test_build_script_and_workflow_stage_and_log_the_switch():
    build = (MOBILE / "ci" / "wheels" / "build_wheels.sh").read_text(encoding="utf-8")
    assert 'python "$wheels_py" stage-ane --platform ios --xcframework "$xcf" --forge "$FORGE_DIR" "${packages[@]}"' \
        in build
    # after setup.sh (the support tree exists), before forge builds the recipes
    assert build.index("source ./setup.sh") < build.index("stage-ane --platform ios") < build.index('forge "$FORGE_HOST"')
    workflow = MOBILE.parents[1] / ".github" / "workflows" / "mobile-wheels.yml"
    if not workflow.is_file():
        pytest.skip("not a full repository checkout")
    text = workflow.read_text(encoding="utf-8")
    assert "grep -E '^ane_enabled=' \"$GITHUB_OUTPUT\"" in text and "iOS NPU backend (wheelhouse.toml [ane])" in text


# ---- the recipe and the patch ---------------------------------------------------------------------------------------

def test_recipe_applies_both_patches_and_loads_the_ane_cache_on_ios():
    for sdk, arch in (("iphoneos", "arm64"), ("iphonesimulator", "arm64"), ("iphonesimulator", "x86_64"),
                      ("android", "arm64-v8a"), ("android", "x86_64")):
        meta = _render_recipe(RECIPE, sdk, arch)
        assert meta["patches"] == ["mobile.patch", "ane-backend.patch"], meta["patches"]
        args = meta["build"]["script_env"]["CMAKE_ARGS"].split()
        assert not any("GGML_ANE" in a for a in args)                   # the switch lives in the cache only
        if sdk != "android":
            assert "-C{HOST_PYTHON_HOME}/glossarion/ane-init.cmake" in args
            assert "-DGGML_METAL=ON" in args and meta["build"]["script_env"]["IPHONEOS_DEPLOYMENT_TARGET"] == "13.0"
            # the only placeholder is the one forge provides (checked against forge itself by stage-ane --forge)
            assert set(re.findall(r"\{(\w+)\}", meta["build"]["script_env"]["CMAKE_ARGS"])) <= \
                {"HOST_PYTHON_HOME", "CROSS_VENV_PYTHON"}, args
    assert (RECIPE.parent / "patches" / "ane-backend.patch").is_file()


def _added(body: str) -> list:
    return [line[1:] for line in body.splitlines() if line.startswith("+") and not line.startswith("+++")]


def test_ane_patch_header_names_its_source_and_every_change():
    header = ANE_PATCH.read_text(encoding="utf-8").split("diff --git", 1)[0]
    for needle in ("llama.cpp pull request 31969", "ggml: (research) add ANE (Apple Neural Engine) backend",
                   "xsn/ggml_ane", "a45d7d2e", "DRAFT and NOT MERGED", "MIT", "152 commits older",
                   "a956a852c9c8bc1fb07206364838ddef299178c0", "PRIVATE AppleNeuralEngine.framework",
                   "GLOSSARION_GGML_EAGER_ANE", "llama_cpp_python_install_target(ggml-ane)", "npu-backend.txt",
                   "glossarion_ane_selftest", "the freeze", "<ios18>"):
        assert needle in header, needle
    assert re.findall(r"^(\d)\. ", header, re.M) == ["1", "2", "3", "4", "5", "6", "7", "8"]
    # forge applies it with patch -p1 --ignore-whitespace: nothing in the prose may look like a diff line
    assert not any(line.startswith(("--- ", "+++ ", "@@", "***", "Index:")) for line in header.splitlines())


def test_ane_patch_files_and_the_glossarion_changes():
    hunks = _file_diffs(ANE_PATCH.read_text(encoding="utf-8"))
    assert sorted(hunks) == sorted([
        "CMakeLists.txt", "vendor/llama.cpp/ggml/CMakeLists.txt", "vendor/llama.cpp/ggml/include/ggml-ane.h",
        "vendor/llama.cpp/ggml/src/CMakeLists.txt", "vendor/llama.cpp/ggml/src/ggml-ane/CMakeLists.txt",
        "vendor/llama.cpp/ggml/src/ggml-ane/ggml-ane-mil.cpp", "vendor/llama.cpp/ggml/src/ggml-ane/ggml-ane-mil.h",
        "vendor/llama.cpp/ggml/src/ggml-ane/ggml-ane-runtime.h", "vendor/llama.cpp/ggml/src/ggml-ane/ggml-ane-runtime.m",
        "vendor/llama.cpp/ggml/src/ggml-ane/ggml-ane.cpp", "vendor/llama.cpp/ggml/src/ggml-backend-reg.cpp"])
    # no Python here (the loader is mobile.patch 11.), and the pull request's test tools are not taken
    assert not any(name.endswith(".py") or "test-backend-ops" in name or "scripts/apple" in name for name in hunks)
    # 1. the registry registers the ANE only with GLOSSARION_GGML_EAGER_ANE (its device count dlopens the framework)
    reg = _added(hunks["vendor/llama.cpp/ggml/src/ggml-backend-reg.cpp"])
    assert '#include "ggml-ane.h"' in reg
    # with GGML_ANE OFF (Android, [ane] switched off) the patch adds no code: every added C++ line of the registry
    # sits under #ifdef GGML_USE_ANE (no load_all_from_path line), and the header is not installed (8.)
    assert not any("load_best" in line for line in reg)
    code = [line for line in reg if line.strip() and not line.strip().startswith("//")]
    assert code[0] == "#ifdef GGML_USE_ANE" and code.count("#ifdef GGML_USE_ANE") == code.count("#endif") == 2
    assert "ggml-ane.h" not in "\n".join(_added(hunks["vendor/llama.cpp/ggml/CMakeLists.txt"]))
    i = reg.index("            register_backend(ggml_backend_ane_reg());")
    assert reg[i - 1] == '        if (getenv("GLOSSARION_GGML_EAGER_ANE") != nullptr) {' and reg[i + 1] == "        }"
    assert reg.count("            register_backend(ggml_backend_ane_reg());") == 1
    assert not any(line == "        register_backend(ggml_backend_ane_reg());" for line in reg)
    # 2. installed with the other libraries
    assert "    llama_cpp_python_install_target(ggml-ane)" in _added(hunks["CMakeLists.txt"])
    # the switch: an option that defaults to OFF, the backend added by ggml_add_backend(ANE)
    assert 'option(GGML_ANE                             "ggml: use the Apple Neural Engine (experimental)" OFF)' in \
        _added(hunks["vendor/llama.cpp/ggml/CMakeLists.txt"])
    assert "ggml_add_backend(ANE)" in _added(hunks["vendor/llama.cpp/ggml/src/CMakeLists.txt"])
    # 3. the marker, the CPU backend for the freeze, ARC for the Objective-C file, the two public frameworks
    cmake = "\n".join(_added(hunks["vendor/llama.cpp/ggml/src/ggml-ane/CMakeLists.txt"]))
    assert 'file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/npu-backend.txt "libggml-ane.dylib\\n")' in cmake
    assert "install(FILES ${CMAKE_CURRENT_BINARY_DIR}/npu-backend.txt DESTINATION llama_cpp)" in cmake
    assert "target_link_libraries(ggml-ane PRIVATE ggml-cpu)" in cmake and "GLOSSARION_ANE_CPU_FALLBACK" in cmake
    assert 'set_source_files_properties(ggml-ane-runtime.m PROPERTIES COMPILE_FLAGS "-fobjc-arc")' in cmake
    assert "find_library(FOUNDATION_LIBRARY Foundation REQUIRED)" in cmake and \
        "find_library(IOSURFACE_LIBRARY  IOSurface  REQUIRED)" in cmake
    sources = cmake.split("ggml_add_backend_library(ggml-ane", 1)[1].split(")", 1)[0].split()
    assert sorted(sources) == ["ggml-ane-mil.cpp", "ggml-ane-runtime.m", "ggml-ane.cpp"]
    # 4. the compile counters
    runtime = "\n".join(_added(hunks["vendor/llama.cpp/ggml/src/ggml-ane/ggml-ane-runtime.m"]))
    assert "#error \"ggml-ane-runtime.m must be compiled with -fobjc-arc\"" in runtime    # ARC proven at build time
    for counter in ("g_cnt_fresh", "g_cnt_cached", "g_cnt_bytes", "g_cnt_failures"):
        assert f"atomic_fetch_add(&{counter}" in runtime, counter
    assert 'dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)' in runtime
    # 5./6. the pending counts, the freeze and its CPU fallback
    cpp = "\n".join(_added(hunks["vendor/llama.cpp/ggml/src/ggml-ane/ggml-ane.cpp"]))
    assert "if (g_glossarion_frozen && !g_ane_collect) {" in cpp
    assert "return g_ane_collect ? GGML_STATUS_SUCCESS : glossarion_run_on_cpu(ctx, cgraph, i0, i1);" in cpp
    assert "if (e->fam && e->fam->n_compiled < e->fam->funcs.size() && !g_glossarion_frozen) {" in cpp
    assert "ctx->glossarion_cpu = ggml_backend_cpu_init();" in cpp and "ggml_backend_graph_compute(ctx->glossarion_cpu" in cpp
    assert "ggml_graph_view(" not in cpp.replace("ggml_graph_view() returns", "")  # not exported by ggml-base
    # the backend's own development switches stay as they are (the loader removes them from the environment)
    assert 'getenv("GGML_ANE_PURGE_CACHE")' in runtime


def test_ane_header_declares_what_the_loader_calls():
    hunks = _file_diffs(ANE_PATCH.read_text(encoding="utf-8"))
    header = "\n".join(_added(hunks["vendor/llama.cpp/ggml/include/ggml-ane.h"]))
    for name in EXPORTS:
        assert re.search(r"GGML_BACKEND_API\s+\S+(\s+\S+)?\s+" + name + r"\(", header), name
    cpp = "\n".join(_added(hunks["vendor/llama.cpp/ggml/src/ggml-ane/ggml-ane.cpp"]))
    for name in EXPORTS:
        assert re.search(r"^\S[^(]*\b" + name + r"\(", cpp, re.M), name                # defined
    assert tuple(EXPORTS) == wh.ANE_EXPORTS
    # struct glossarion_ane_stats: the loader's ctypes mirror has the same fields, types and order
    body = re.search(r"struct glossarion_ane_stats \{\n(.*?)\n\};", header, re.S).group(1)
    fields = [(t, n) for t, n in re.findall(r"^\s*(int64_t|int32_t)\s+(\w+);", body, re.M)]
    ns = _patched_cpu_loader(_real_ctypes_ns())
    assert [n for _t, n in fields] == list(ns["ANE_STATS_FIELDS"])
    assert [t for t, _n in fields] == ["int64_t"] * 7 + ["int32_t"]
    stats = ns["_ane_stats_type"]()
    assert [(f, t) for f, t in stats._fields_] == [(n, real_ctypes_type(t)) for t, n in fields]


def real_ctypes_type(name: str):
    import ctypes

    return {"int64_t": ctypes.c_int64, "int32_t": ctypes.c_int32}[name]


def test_mobile_patch_header_describes_the_ios_loader():
    header = PATCH.read_text(encoding="utf-8").split("diff --git", 1)[0]
    assert "11. Optional NPU backend on iOS (Apple Neural Engine" in header
    assert "patches/ane-backend.patch" in header and "GGML_ANE_PURGE_CACHE" in header
    loader = _file_diffs(PATCH.read_text(encoding="utf-8"))["llama_cpp/_ctypes_extensions.py"]
    # the loader never sets the eager switch or the purge (only names them in comments / the removal list)
    assert 'os.environ["GLOSSARION_GGML_EAGER_ANE"]' not in loader and 'os.environ["GGML_ANE_PURGE_CACHE"]' not in loader
    assert '"GGML_ANE_PURGE_CACHE", "GGML_ANE_DUMP_DIR", "GGML_ANE_PROFILE"' in loader


# ---- verify-native / verify-site-packages ---------------------------------------------------------------------------

ANE_FRAMEWORKS = ("/System/Library/Frameworks/Foundation.framework/Foundation",
                  "/System/Library/Frameworks/IOSurface.framework/IOSurface")
ANE_NEEDED = {**IOS_LLAMA_NEEDED, "libggml-ane": ("libggml-base.so", "libggml-cpu.so"),
              "libggml": ("libggml-cpu.so", "libggml-metal.so", "libggml-ane.so", "libggml-base.so")}


def make_ane_ios(sp: Path, target: str, libs=IOS_LLAMA_LIBS + ("libggml-ane",), marker="libggml-ane.dylib",
                 needed=None, extra_dylibs=None, exports=EXPORTS, minos=None) -> Path:
    """build/site-packages/<target> with the iOS engine, libggml-ane (linked into libggml, its exports defined) and
    the marker. ``needed``: {stem: sibling list} replacing ANE_NEEDED's; ``extra_dylibs``: {stem: more dylibs}."""
    arch = sp / target
    _llama_dist_info(arch)
    lib = arch / "llama_cpp" / "lib"
    lib.mkdir(parents=True)
    cpu, plat, default_minos = IOS_SLICES[target]
    deps = {**ANE_NEEDED, **(needed or {})}
    for stem in libs:
        dylibs = [f"@rpath/{d[:-3]}.dylib" for d in deps[stem]] + ["/usr/lib/libc++.1.dylib", "/usr/lib/libSystem.B.dylib"]
        if stem == "libggml-metal":
            dylibs += list(METAL_FRAMEWORKS)
        if stem == "libggml-ane":
            dylibs += ["/usr/lib/libobjc.A.dylib", *ANE_FRAMEWORKS]
        dylibs += list((extra_dylibs or {}).get(stem, ()))
        imports = [("_malloc", len(dylibs)), ("__ZNSt3__14cout", 3 if len(dylibs) > 2 else 1)]
        defined = [f"_{s}" for s in exports] if stem == "libggml-ane" else [f"_{stem[3:]}_api"]
        (lib / f"{stem}.dylib").write_bytes(make_macho(cputype=cpu, platform=plat, minos=minos or default_minos,
                                                       dylibs=tuple(dylibs), imports=imports, defined=defined))
    if marker is not None:
        (arch / "llama_cpp" / "npu-backend.txt").write_text(marker + "\n", encoding="utf-8")
    return arch


def test_parse_macho_reads_the_defined_externals():
    macho = wh.parse_macho(make_macho(defined=("_glossarion_ane_probe", "_x"), imports=[("_malloc", 2)]))
    assert macho["defined"] == ["_glossarion_ane_probe", "_x"] and macho["undefined"] == [("_malloc", 2)]


def test_verify_site_packages_accepts_the_ane_backend(tmp_path):
    for target in IOS_SLICES:
        make_ane_ios(tmp_path, target)
    errors, notes = _verify_llama(tmp_path, "ios", ane=wh.ANE_TARGETS)
    assert errors == []
    assert "iphoneos.arm64: llama_cpp/npu-backend.txt marks the NPU backend; no HTP skeleton ships" in notes
    assert any("lib/libggml-ane.dylib iOS NPU backend: linked into libggml" in n for n in notes)


@pytest.mark.parametrize("kwargs, message", [
    ({"libs": IOS_LLAMA_LIBS, "needed": {"libggml": IOS_LLAMA_NEEDED["libggml"]}},
     "expected exactly ['libggml', 'libggml-ane', 'libggml-base'"),
    ({"marker": None}, "iphoneos.arm64: no llama_cpp/npu-backend.txt (the NPU backend's marker, patches/ane-backend.patch)"),
    ({"marker": "libggml-hexagon.so"}, "says 'libggml-hexagon.so', expected 'libggml-ane.dylib'"),
    ({"needed": {"libggml": IOS_LLAMA_NEEDED["libggml"]}}, "lib/libggml.dylib does not link @rpath/libggml-ane.dylib"),
    ({"needed": {"libggml-ane": ("libggml-base.so", "libllama.so")}},
     "lib/libggml-ane.dylib: links ['@rpath/libllama.dylib'] (only libggml-base, libggml-cpu of its siblings)"),
    ({"extra_dylibs": {"libggml-ane": ("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",)}},
     "a private framework may only be loaded at run time (dlopen), never linked"),
    ({"extra_dylibs": {"libllama": ("/System/Library/PrivateFrameworks/X.framework/X",)}},
     "lib/libllama.dylib: links ['/System/Library/PrivateFrameworks/X.framework/X']: a private framework"),
    ({"extra_dylibs": {"libggml-ane": ("/System/Library/Frameworks/Metal.framework/Metal",)}},
     "lib/libggml-ane.dylib: links ['/System/Library/Frameworks/Metal.framework/Metal'] (only its siblings"),
    ({"exports": EXPORTS[:4] + EXPORTS[5:]}, "lib/libggml-ane.dylib: does not export ['glossarion_ane_selftest']"),
    ({"minos": (18, 0, 0)}, "minos 18.0.0 (LC_BUILD_VERSION) is above 13.0"),
])
def test_verify_site_packages_rejects_ane_mistakes(tmp_path, kwargs, message):
    make_ane_ios(tmp_path, "iphoneos.arm64", **kwargs)
    errors, _ = _verify_llama(tmp_path, "ios", ane=wh.ANE_TARGETS)
    assert any(message in e for e in errors), errors


def test_switched_off_ane_expects_no_backend(tmp_path):
    make_ane_ios(tmp_path / "on", "iphoneos.arm64")
    errors, _ = _verify_llama(tmp_path / "on", "ios", ane=())
    text = "\n".join(errors)
    assert "llama_cpp/npu-backend.txt is there, but no NPU backend is expected ([ane] is switched off)" in text
    assert "expected exactly ['libggml', 'libggml-base', 'libggml-cpu', 'libggml-metal', 'libllama', 'libmtmd']" in text
    # the plain Metal build passes switched off, and is missing the backend when [ane] is on
    base.make_llama_ios(tmp_path / "off", "iphoneos.arm64")
    assert _verify_llama(tmp_path / "off", "ios", ane=())[0] == []
    errors, _ = _verify_llama(tmp_path / "off", "ios", ane=wh.ANE_TARGETS)
    assert any("no llama_cpp/npu-backend.txt" in e for e in errors) and any("libggml-ane" in e for e in errors)


def test_android_marker_must_name_the_hexagon_library(tmp_path):
    from test_engine_npu_build import HEXAGON, make_npu_android
    from test_check_mobile_wheels_wheelhouse import LLAMA_NDK_EXPORTS, make_ndk

    arch = make_npu_android(tmp_path / "sp")
    (arch / "llama_cpp" / "npu-backend.txt").write_text("libggml-ane.dylib\n", encoding="utf-8")
    errors, _ = _verify_llama(tmp_path / "sp", "android", make_ndk(tmp_path / "ndk", LLAMA_NDK_EXPORTS), hexagon=HEXAGON)
    assert any("says 'libggml-ane.dylib', expected 'libggml-hexagon.so'" in e for e in errors), errors


def test_verify_native_checks_the_ane_backend_of_the_wheel(tmp_path):
    wheel = _zip_tree(make_ane_ios(tmp_path / "sp", "iphoneos.arm64"),
                      tmp_path / "llama_cpp_python-0.3.37-1000-cp313-cp313-ios_13_0_arm64_iphoneos.whl")
    errors, notes = wh.verify_native_wheel(wheel, "llama-cpp-python", "iphoneos.arm64", "3.5.9", (13, 0),
                                           None, {}, wh.ANE_TARGETS)
    assert errors == [], errors
    assert any("llama_cpp/npu-backend.txt marks the NPU backend" in n for n in notes)
    errors, _ = wh.verify_native_wheel(wheel, "llama-cpp-python", "iphoneos.arm64", "3.5.9", (13, 0), None, {}, ())
    assert any("([ane] is switched off)" in e for e in errors), errors
    # a wheel whose libggml-ane lacks the Glossarion functions (the bare pull request) fails
    bare = _zip_tree(make_ane_ios(tmp_path / "sp2", "iphoneos.arm64", exports=EXPORTS[:3]), tmp_path / "bare.whl")
    errors, _ = wh.verify_native_wheel(bare, "llama-cpp-python", "iphoneos.arm64", "3.5.9", (13, 0), None, {},
                                       wh.ANE_TARGETS)
    assert any("does not export ['glossarion_ane_probe'" in e for e in errors), errors


# ---- the simulator self-test ----------------------------------------------------------------------------------------

def _ios_report(npu=None, status="pass") -> dict:
    report = base.selftest_report(platform="ios")
    report["checks"].append({"name": "llama_engine", "status": status, "detail": {"version": "0.3.37", "npu": npu}})
    return report


def _npu_entry(shipped=True, device_ok=False, framework_ok=False, functions=True, precheck=True,
               registered=()) -> dict:
    steps = [{"name": "shipped", "ok": shipped, "detail": "libggml-ane.dylib"},
             {"name": "device", "ok": device_ok, "detail": "iPhone simulator: no Neural Engine"},
             {"name": "framework", "ok": framework_ok, "detail": "not checked (an earlier step failed)"}]
    entry = {"functions": functions, "shipped": shipped, "platform": "ios", "device_index": True,
             "registered": list(registered), "requested": False}
    if precheck:
        entry["precheck"] = {"backend": "ANE", "arch": None, "chip": None, "error": steps[1]["detail"], "steps": steps}
    return entry


def test_simulator_self_test_shows_the_backend_and_stops_at_the_device_step():
    assert wh.check_ane_selftest(_ios_report(_npu_entry()), True) == []
    assert wh.check_ane_selftest(_ios_report(_npu_entry(shipped=False, precheck=False)), False) == []
    cases = [
        (_ios_report(_npu_entry(shipped=False)), True, "shipped = False, but wheelhouse.toml [ane] is enabled"),
        (_ios_report(_npu_entry()), False, "shipped = True, but wheelhouse.toml [ane] is switched off"),
        (_ios_report(_npu_entry(functions=False)), True, "the loader's NPU functions are missing"),
        (_ios_report(_npu_entry(precheck=False)), True, "no precheck steps"),
        (_ios_report(_npu_entry(device_ok=True)), True, "step 'device' should fail on the simulator"),
        (_ios_report(_npu_entry(device_ok=True, framework_ok=True)), True, "step 'framework' ran on the simulator"),
        # registered at run time without the app asking (the registry's start, an env switch)
        (_ios_report(_npu_entry(registered=["ANE"])), True, "registered = ['ANE'], expected []"),
        (_ios_report({k: v for k, v in _npu_entry().items() if k != "registered"}), True, "registered = None"),
        (_ios_report(None), True, "llama_engine has no npu entry"),
        (_ios_report(_npu_entry(), status="fail"), True, "check 'llama_engine' is fail"),
        (base.selftest_report(platform="ios"), True, "check 'llama_engine' is missing from the report"),
    ]
    for report, enabled, message in cases:
        errors = wh.check_ane_selftest(report, enabled)
        assert any(message in e for e in errors), (message, errors)


def test_a_wheels_line_without_the_engine_check_only_warns():
    report = base.selftest_report(platform="ios")
    errors, warning = wh.ane_selftest_errors(report, "console.log (GLOSSARION_WHEELS line, 1 found)", True)
    assert errors == [] and "NPU backend not checked" in warning
    errors, warning = wh.ane_selftest_errors(report, "selftest-smoke.json (self-test report)", True)
    assert warning is None and errors == ["check 'llama_engine' is missing from the report (the engine and its NPU entry)"]
    errors, warning = wh.ane_selftest_errors(_ios_report(_npu_entry()), "x (GLOSSARION_WHEELS line, 1 found)", True)
    assert errors == [] and warning is None


# ---- the IPA --------------------------------------------------------------------------------------------------------

def _verify_ipa():
    spec = importlib.util.spec_from_file_location("glossarion_verify_ipa", MOBILE / "ci" / "verify_ipa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_ipa_checks_the_ane_backend(tmp_path):
    vi = _verify_ipa()
    assert vi.ane_enabled(vi.DEFAULT_WHEELHOUSE) is True and vi.DEFAULT_WHEELHOUSE == wh.DEFAULT_MANIFEST
    assert vi.ane_enabled(_manifest(tmp_path, enabled=False)) is False and vi.ane_enabled(None) is False
    shipped = ["Frameworks/llama_cpp.lib.libggml-ane.framework/llama_cpp.lib.libggml-ane",
               "app/site-packages/llama_cpp/lib/libggml-ane.fwork", "app/site-packages/llama_cpp/npu-backend.txt",
               "app/site-packages/llama_cpp/__init__.py"]
    errors, warnings, facts = vi.ane_problems(shipped, True)
    assert errors == [] and warnings == [] and "enabled" in facts["NPU backend (wheelhouse.toml [ane])"]
    # serious-python may spell the framework with underscores
    assert vi.ane_problems(["Frameworks/llama_cpp_lib_libggml_ane.framework/x"] + shipped[2:], True)[0] == []
    plain = ["Frameworks/llama_cpp.lib.libggml.framework/llama_cpp.lib.libggml", "app/site-packages/llama_cpp/x.py"]
    errors, _w, _f = vi.ane_problems(plain, True)
    assert any("no framework made from libggml-ane" in e for e in errors) and any("no llama_cpp/npu-backend.txt" in e
                                                                                for e in errors)
    assert vi.ane_problems(plain, False)[0] == []
    errors, _w, _f = vi.ane_problems(shipped, False)
    assert len(errors) == 2 and all("switched off" in e for e in errors)
    private = vi.ane_problems(shipped + ["Frameworks/AppleNeuralEngine.framework/AppleNeuralEngine"], True)[0]
    assert any("Apple's private AppleNeuralEngine framework may not ship" in e for e in private)
    # no llama_cpp package visible at all: a warning, not a failure
    errors, warnings, _f = vi.ane_problems(["Frameworks/App.framework/App"], True)
    assert errors == [] and any("not checked" in w for w in warnings)


def _profile(**plist) -> bytes:
    """An embedded.mobileprovision: a CMS envelope (fake bytes) around the XML plist."""
    import plistlib

    return b"0\x82\x1b\x00junk" + plistlib.dumps({"Name": "p", **plist}) + b"\x00\xa0trailer"


def test_verify_ipa_fails_a_store_build_that_carries_the_ane(tmp_path, capsys):
    vi = _verify_ipa()
    assert vi.profile_kind(_profile()) == "store"
    assert vi.profile_kind(_profile(ProvisionedDevices=["x"])) == "ad-hoc"
    assert vi.profile_kind(_profile(ProvisionedDevices=["x"], Entitlements={"get-task-allow": True})) == "development"
    assert vi.profile_kind(_profile(ProvisionsAllDevices=True)) == "enterprise"
    assert vi.profile_kind(b"not a profile") is None

    def ipa(name: str, profile: bytes) -> Path:
        path = tmp_path / f"{name}.ipa"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("Payload/Runner.app/Info.plist", __import__("plistlib").dumps({"CFBundleIdentifier": "x"}))
            z.writestr("Payload/Runner.app/_CodeSignature/CodeResources", b"x")
            z.writestr("Payload/Runner.app/embedded.mobileprovision", profile)
            z.writestr("Payload/Runner.app/app/site-packages/llama_cpp/npu-backend.txt", "libggml-ane.dylib\n")
            z.writestr("Payload/Runner.app/Frameworks/llama_cpp.lib.libggml-ane.framework/llama_cpp.lib.libggml-ane",
                       b"")
        return path

    store_msg = "an App Store / TestFlight build"
    for name, profile, method, fails in (("adhoc", _profile(ProvisionedDevices=["x"]), "release-testing", False),
                                         ("method", _profile(ProvisionedDevices=["x"]), "app-store-connect", True),
                                         ("store", _profile(), "release-testing", True),
                                         ("old", _profile(), "app-store", True)):
        vi.main([str(ipa(name, profile)), "--expect-signed", "--export-method", method])
        out = capsys.readouterr().out
        assert (store_msg in out and "ERROR" in out.split(store_msg)[0].rsplit("\n", 1)[-1]) is fails, (name, out)
        if not fails:
            assert "warning: a signed build carries the iOS NPU backend" in out
    # [ane] switched off: no store complaint at all
    vi.main([str(ipa("off", _profile())), "--expect-signed", "--export-method", "app-store",
             "--wheelhouse", str(_manifest(tmp_path, enabled=False))])
    assert store_msg not in capsys.readouterr().out
    # the workflow passes the signed job's export method
    workflow = MOBILE.parents[1] / ".github" / "workflows" / "build-mobile.yml"
    if workflow.is_file():
        text = workflow.read_text(encoding="utf-8")
        assert '--pyproject src/mobile/pyproject.toml --expect-signed\n          --export-method ' \
               '"${{ steps.signing.outputs.export_method }}"' in text.replace("\r\n", "\n")


def test_verify_ipa_looks_inside_packed_site_packages(tmp_path):
    vi = _verify_ipa()
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("llama_cpp/npu-backend.txt", "libggml-ane.dylib\n")
        z.writestr("llama_cpp/lib/libggml-ane.fwork", "Frameworks/x.framework/x")
    ipa = tmp_path / "app.ipa"
    with zipfile.ZipFile(ipa, "w") as z:
        z.writestr("Payload/Runner.app/Info.plist", b"")
        z.writestr("Payload/Runner.app/app/app.zip", inner.getvalue())
        z.writestr("Payload/Runner.app/Frameworks/llama_cpp.lib.libggml-ane.framework/llama_cpp.lib.libggml-ane", b"")
    paths = vi.bundle_paths(vi.Bundle(ipa))
    assert "app/app.zip!llama_cpp/npu-backend.txt" in paths
    assert vi.ane_problems(paths, True)[0] == []


# ---- the patched loader on iOS over a fake ctypes -------------------------------------------------------------------

IOSInfo = collections.namedtuple("IOSInfo", "system release model is_simulator")


def _ane_loader(tmp_path, monkeypatch, *, marker="libggml-ane.dylib", lib=True, missing=(), release="18.2",
                model="iPhone15,2", simulator=False, probe=1, chip=b"h15", reg=0x51, devices=("ANE",), selftest=0,
                cached=0, libggml=True):
    """The loader with a fake libggml-ane and libggml, a marker in a fake package and platform.ios_ver() faked."""
    for key in ANE_ENV:
        monkeypatch.delenv(key, raising=False)
    calls = {"probe": 0, "reg": [], "register": [], "selftest": 0, "kv": [], "threads": [], "freeze": [],
             "collect": [], "reset": 0}

    def probe_fn(buf, size):
        calls["probe"] += 1
        if probe == 1:
            buf.value = chip
        return probe

    def reg_fn():
        calls["reg"].append({k: os.environ.get(k) for k in ANE_ENV})
        return reg

    def selftest_fn(err, size, cached_ref):
        calls["selftest"] += 1
        cached_ref._obj.value = cached
        if selftest:
            err.value = b"test program: compile failed: the ANE compiler said no"
        return selftest

    def stats_fn(ref):
        for i, name in enumerate(("families_pending", "bytes_pending", "compiles_fresh", "compiles_cached",
                                  "bytes_fresh", "compile_failures", "cpu_segments", "storage_refused")):
            setattr(ref._obj, name, i + 1)

    def reset_fn():
        calls["reset"] += 1

    functions = {
        "ggml_backend_ane_reg": reg_fn, "glossarion_ane_probe": probe_fn, "glossarion_ane_selftest": selftest_fn,
        "ggml_backend_ane_set_kv_buckets": lambda arr, n: calls["kv"].append(list(arr)[:n]),
        "glossarion_ane_set_cpu_threads": lambda n: calls["threads"].append(n),
        "glossarion_ane_freeze": lambda on: calls["freeze"].append(on),
        "ggml_backend_ane_collect_shapes": lambda on: calls["collect"].append(on),
        "glossarion_ane_get_stats": stats_fn, "glossarion_ane_reset_stats": reset_fn}
    ns = _patched_cpu_loader(_real_ctypes_ns())
    pkg = tmp_path / "pkg" / "llama_cpp"
    pkg.mkdir(parents=True, exist_ok=True)
    if marker is not None:
        (pkg / "npu-backend.txt").write_text(marker + "\n", encoding="utf-8")
    ns["_NPU_MARKER_SOURCE"] = pkg / "npu-backend.txt"
    ns["_LIBGGML_ANE"] = types.SimpleNamespace(**{k: _FakeFn(f) for k, f in functions.items() if k not in missing}) \
        if lib else None
    ns["_LIBGGML"] = types.SimpleNamespace(
        ggml_backend_register=_FakeFn(lambda r: calls["register"].append(r)),
        ggml_backend_reg_dev_count=_FakeFn(lambda r: len(devices)),
        ggml_backend_reg_dev_get=_FakeFn(lambda r, i: 200 + i),
        ggml_backend_dev_name=_FakeFn(lambda dev: devices[dev - 200].encode())) if libggml else None
    monkeypatch.setattr(platform, "ios_ver", lambda: IOSInfo("iOS", release, model, simulator), raising=False)
    return ns, calls


def test_ios_loader_names_the_backend_from_its_marker(tmp_path, monkeypatch):
    ns, calls = _ane_loader(tmp_path, monkeypatch)
    assert ns["npu_backend_name"]() == "ANE" and ns["npu_backend_shipped"]() is True
    assert calls["probe"] == 0 and calls["reg"] == []                              # naming it calls nothing
    assert ns["npu_skeleton_source"]() == {"release": RELEASE, "commit": COMMIT, "url": "", "size": 0, "sha256": "",
                                           "files": {}}
    ns["npu_skeleton_source"]()["files"]["x"] = 1
    assert ns["npu_skeleton_source"]()["files"] == {}                              # a copy
    for text, name in (("libggml-hexagon.so", "HTP"), ("something else", None)):
        (tmp_path / "pkg" / "llama_cpp" / "npu-backend.txt").write_text(text, encoding="utf-8")
        assert ns["npu_backend_name"]() == name
    (tmp_path / "pkg" / "llama_cpp" / "npu-backend.txt").unlink()
    assert ns["npu_backend_name"]() is None
    # the ANE marker, but no libggml-ane loaded (or without its registry function): not shipped
    ns, _calls = _ane_loader(tmp_path / "b", monkeypatch, lib=False)
    assert ns["npu_backend_shipped"]() is False
    ns, _calls = _ane_loader(tmp_path / "c", monkeypatch, missing=("ggml_backend_ane_reg",))
    assert ns["npu_backend_shipped"]() is False


@pytest.mark.parametrize("kw, steps, detail", [
    ({}, [("shipped", True), ("device", True), ("framework", True)], "Apple Neural Engine h15"),
    ({"simulator": True, "model": "iPhone"}, [("shipped", True), ("device", False), ("framework", False)],
     "iPhone simulator: no Neural Engine"),
    ({"release": "17.5"}, [("shipped", True), ("device", False), ("framework", False)],
     "needs iOS 18 or later (this phone: iOS 17.5)"),
    ({"release": ""}, [("shipped", True), ("device", False), ("framework", False)], "not an iPhone or iPad"),
    ({"lib": False}, [("shipped", False), ("device", False), ("framework", False)],
     "this build carries no NPU backend (libggml-ane)"),
    ({"probe": 0}, [("shipped", True), ("device", True), ("framework", False)],
     "the Apple Neural Engine framework did not load"),
    ({"missing": ("glossarion_ane_probe",)}, [("shipped", True), ("device", True), ("framework", False)],
     "not checked: this build has no framework check"),
])
def test_ios_precheck_checks_the_device_before_the_framework(tmp_path, monkeypatch, kw, steps, detail):
    ns, calls = _ane_loader(tmp_path, monkeypatch, **kw)
    for _ in range(2):                                                            # any number of calls
        result = ns["npu_precheck"]()
        assert _steps(result) == steps, result
        assert set(result) == {"backend", "arch", "chip", "error", "steps"} and result["backend"] == "ANE"
        assert result["arch"] is None                                             # no Hexagon arch rule applies
        last = next((s for s in result["steps"] if not s["ok"]), result["steps"][-1])
        assert detail in last["detail"]
        assert result["chip"] == ("h15" if all(ok for _n, ok in steps) else None)
    # the framework is touched only after the device step passed; nothing is registered or set
    device_ok = dict(steps)["device"]
    assert calls["probe"] == (2 if device_ok and "missing" not in kw else 0)
    assert calls["reg"] == [] and calls["register"] == [] and ns["NPU_STATE"] == {}
    assert all(os.environ.get(k) is None for k in ANE_ENV)
    assert ns["NPU_PRECHECK_STEPS"] == ("shipped", "libcdsprpc", "arch")         # Android's, unchanged


def test_ios_register_sets_the_environment_registers_and_compiles_once(tmp_path, monkeypatch):
    ns, calls = _ane_loader(tmp_path, monkeypatch)
    for key in ("GGML_ANE_PURGE_CACHE", "GGML_ANE_DUMP_DIR", "GGML_ANE_PROFILE"):
        monkeypatch.setenv(key, "1")
    state = ns["register_npu_backend"](str(tmp_path / "skel"), {"GGML_ANE_SEG_WEIGHT_MB": 32, "PATH": "/x",
                                                                "GGML_HEXAGON_ARCH": "79"})
    assert _steps(state) == [(n, True) for n in ANE_STEPS], state
    assert state["backend"] == "ANE" and state["device"] == "ANE" and state["chip"] == "h15"
    assert state["arch"] is None and state["error"] == ""
    assert state["steps"][3]["detail"] == ("ANE registered, 1 device: ANE; ignored options ['PATH', "
                                           "'GGML_HEXAGON_ARCH'] (only GGML_ANE_WEIGHTS, GGML_ANE_SEG_WEIGHT_MB)")
    assert state["steps"][4]["detail"] == "test program compiled and ran"
    # the environment when the backend was registered: int8 weights, the accepted option, no development switch
    assert calls["reg"] == [{"GGML_ANE_WEIGHTS": "int8", "GGML_ANE_SEG_WEIGHT_MB": "32", "GGML_ANE_PURGE_CACHE": None,
                             "GGML_ANE_DUMP_DIR": None, "GGML_ANE_PROFILE": None, "GLOSSARION_GGML_EAGER_ANE": None}]
    assert calls["register"] == [0x51] and calls["selftest"] == 1 and os.environ.get("GGML_HEXAGON_ARCH") is None
    assert not (tmp_path / "skel").exists()                                       # skel_dir is ignored
    # remembered for the process: no second registration or compile
    again = ns["register_npu_backend"](None, {"GGML_ANE_WEIGHTS": "fp16"})
    assert again == state and calls["register"] == [0x51] and calls["selftest"] == 1
    assert os.environ["GGML_ANE_WEIGHTS"] == "int8"


def test_ios_register_reports_a_cache_hit_and_keeps_the_callers_weights(tmp_path, monkeypatch):
    ns, calls = _ane_loader(tmp_path, monkeypatch, cached=1)
    state = ns["register_npu_backend"]("", {"GGML_ANE_WEIGHTS": "fp16"})
    assert state["steps"][4] == {"name": "compile", "ok": True, "detail": "test program ran (from the iOS cache)"}
    assert calls["reg"][0]["GGML_ANE_WEIGHTS"] == "fp16"


@pytest.mark.parametrize("kw, failed, detail, remembered", [
    ({"simulator": True}, "device", "iPhone simulator: no Neural Engine", False),
    ({"probe": 0}, "framework", "the Apple Neural Engine framework did not load", False),
    ({"libggml": False}, "registry", "the engine's libraries are not loaded", False),
    ({"reg": 0}, "registry", "the ANE backend did not start", True),
    ({"devices": ()}, "registry", "ANE registered with 0 device(s)", True),
    ({"devices": ("ANE", "ANE1")}, "registry", "ANE registered with 2 device(s): ANE", True),
    ({"selftest": -1}, "compile", "test program: compile failed: the ANE compiler said no", True),
    ({"missing": ("glossarion_ane_selftest",)}, "compile", "not checked: this build has no test program", True),
])
def test_ios_register_reports_the_first_failure_and_never_raises(tmp_path, monkeypatch, kw, failed, detail, remembered):
    ns, calls = _ane_loader(tmp_path, monkeypatch, **kw)
    state = ns["register_npu_backend"](None)
    names = [s["name"] for s in state["steps"]]
    assert names == list(ANE_STEPS)
    first = next(s for s in state["steps"] if not s["ok"])
    assert first["name"] == failed and detail in first["detail"], state
    assert state["error"] == first["detail"]
    assert all(s["detail"] == "not checked (an earlier step failed)" for s in state["steps"][names.index(failed) + 1:])
    if remembered:                      # the backend may be registered: the answer stays, nothing runs twice
        assert ns["NPU_STATE"] == state and len(calls["reg"]) == 1
        assert ns["register_npu_backend"](None) == state and len(calls["reg"]) == 1
    else:                               # stopped before: nothing registered, set or remembered
        assert ns["NPU_STATE"] == {} and calls["reg"] == [] and os.environ.get("GGML_ANE_WEIGHTS") is None


def test_ios_register_turns_an_exception_into_a_step(tmp_path, monkeypatch):
    ns, _calls = _ane_loader(tmp_path, monkeypatch)
    ns["_ios_device"] = lambda: 1 / 0
    state = ns["register_npu_backend"](None)
    bad = next(s for s in state["steps"] if not s["ok"])
    assert bad["name"] == "device" and bad["detail"].startswith("ZeroDivisionError") and ns["NPU_STATE"] == {}
    pre = ns["npu_precheck"]()
    assert _steps(pre) == [("shipped", True), ("device", False), ("framework", False)]


def test_ios_without_a_marker_uses_the_platform(tmp_path, monkeypatch):
    ns, _calls = _ane_loader(tmp_path, monkeypatch, marker=None)
    monkeypatch.setattr(ns["sys"], "platform", "ios")
    result = ns["npu_precheck"]()
    assert result["backend"] == "ANE" and _steps(result)[0] == ("shipped", False)
    assert ns["npu_skeleton_source"]()["files"] == {}


def test_ios_compile_pass_primitives(tmp_path, monkeypatch):
    ns, calls = _ane_loader(tmp_path, monkeypatch)
    assert ns["npu_model_begin"](4096, 4) is None
    assert calls["kv"] == [[4096]] and calls["threads"] == [4] and calls["freeze"] == [False] and calls["reset"] == 1
    ns["npu_collect"](True)
    ns["npu_collect"](False)
    ns["npu_freeze"](True)
    assert calls["collect"] == [True, False] and calls["freeze"] == [False, True]
    assert ns["npu_stats"]() == {"families_pending": 1, "bytes_pending": 2, "compiles_fresh": 3, "compiles_cached": 4,
                                 "bytes_fresh": 5, "compile_failures": 6, "cpu_segments": 7, "storage_refused": 8}
    # elsewhere (no libggml-ane: Android, desktop) they do nothing
    ns["_LIBGGML_ANE"] = None
    assert ns["npu_model_begin"](4096, 4) is None and ns["npu_collect"](True) is None
    assert ns["npu_freeze"](True) is None and ns["npu_stats"]() == {}
    assert calls["kv"] == [[4096]] and calls["collect"] == [True, False]


def test_device_index_finds_the_ane_and_metal_by_registry_name():
    ns = _patched_cpu_loader(_real_ctypes_ns())
    CPU, GPU = 0, 1
    ns["_LIBGGML"] = _device_libggml([("CPU", CPU, None), ("ANE", GPU, None), ("MTL", GPU, None)])
    assert ns["device_index"]("ANE") == 0 and ns["device_index"]("MTL") == 1 and ns["device_index"]("Metal") == -1
