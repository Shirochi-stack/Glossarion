"""The on-device engine's optional NPU backend in the build (wheelhouse.toml [hexagon], recipes/llama-cpp-python,
mobile.patch 10., ci/wheels/wheels.py, ci/verify_apk.py): the manifest switch and the HTP skeleton pins (the same
in the toml and the patched loader), the SDK staging with a tiny fake SDK, the CMake initial cache the arm64 recipe
loads, the verify rules for libggml-hexagon and its marker and against any shipped HTP skeleton (fake arm64 /
QDSP6 ELF files), the patched loader's npu_precheck / register_npu_backend / device_index over a fake ctypes, the
patch's own hunk arithmetic, and the APK checks. No network, no SDK, no native NPU.
"""

from __future__ import annotations

import argparse
import ctypes as real_ctypes
import hashlib
import io
import os
import re
import struct
import tarfile
import types
import zipfile
from pathlib import Path

import pytest

import test_check_mobile_wheels_wheelhouse as base
from test_check_mobile_wheels_wheelhouse import (LLAMA_LIBS_ARM64, LLAMA_NDK_EXPORTS, MOBILE, _FakeFn,
                                                 _patched_cpu_loader, _render_recipe, _verify_llama, _zip_tree,
                                                 make_elf, make_llama_android, make_ndk, wh)
from test_engine_gpu_build import _verify_apk

RECIPE = MOBILE / "ci" / "wheels" / "recipes" / "llama-cpp-python" / "meta.yaml"
PATCH = RECIPE.parent / "patches" / "mobile.patch"
ARCHS = ("v73", "v75", "v79", "v81", "v85")
HEX_LIBS_ARM64 = LLAMA_LIBS_ARM64 + ("libggml-hexagon",)
RELEASE = "b12387"
COMMIT = "a956a852c9c8bc1fb07206364838ddef299178c0"


def _real_ctypes_ns():
    return types.SimpleNamespace(**{k: getattr(real_ctypes, k) for k in dir(real_ctypes) if not k.startswith("__")})


# ---- the manifest switch and the skeleton pins ---------------------------------------------------------------------

def _manifest(tmp_path: Path, enabled: bool = True, sdk_sha: str | None = None, extra: str = "") -> Path:
    """A copy of the real wheelhouse.toml (absolute recipe paths) with [hexagon] enabled or switched off."""
    text = wh.DEFAULT_MANIFEST.read_text(encoding="utf-8")
    recipes = (MOBILE / "ci" / "wheels" / "recipes").as_posix()
    text = text.replace('recipe = "recipes/', f'recipe = "{recipes}/')
    assert "[hexagon]\nenabled = true\n" in text.replace("\r\n", "\n")
    text = text.replace("\r\n", "\n").replace("[hexagon]\nenabled = true\n",
                                               f"[hexagon]\nenabled = {'true' if enabled else 'false'}\n{extra}")
    if sdk_sha:
        text = re.sub(r'(sdk = \{[^}]*sha256 = ")[0-9a-f]{64}', lambda m: m.group(1) + sdk_sha, text)
    path = tmp_path / "manifest" / "wheelhouse.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_manifest_pins_the_sdk_and_switches_the_backend_on_for_android_arm64_only():
    ctx = wh.Ctx(wh.DEFAULT_MANIFEST, "android")
    spec = wh.hexagon_spec(ctx.data)
    assert spec is not None, "[hexagon] stays enabled on this branch"
    assert spec["abis"] == ("arm64-v8a",) and spec["htp_archs"] == ARCHS
    assert spec["version"] == "6.7.0.0" and spec["tools"] == "21.0.04"
    sdk = spec["sdk"]
    assert sdk["sha256"] == "00725fc449916d7b9af2c2a3c6b54d61f3c872d5dbc1d3ca3fd2f839b1d5dc70"
    assert sdk["url"] == ("https://github.com/snapdragon-toolchain/hexagon-sdk/releases/download/v6.7.0.0/"
                          + sdk["file"]) and sdk["file"] == "hexagon-sdk-v6.7.0.0-amd64-lnx.tar.xz"
    assert spec["skeletons"]["release"] == RELEASE and sorted(spec["skeletons"]["files"]) == list(ARCHS)
    assert wh.hexagon_targets(ctx) == {"arm64-v8a": ARCHS}
    assert wh.hexagon_targets(wh.Ctx(wh.DEFAULT_MANIFEST, "ios")) == {"arm64-v8a": ARCHS}   # android targets only
    assert wh.needs_hexagon_sdk(ctx, ["llama-cpp-python"]) and not wh.needs_hexagon_sdk(ctx, ["pillow"])
    assert not wh.needs_hexagon_sdk(wh.Ctx(wh.DEFAULT_MANIFEST, "ios"), ["llama-cpp-python"])


def test_skeleton_pins_are_llama_cpp_releases_files_of_the_vendored_commit():
    pins = wh.hexagon_skeleton_pins(wh.Ctx(wh.DEFAULT_MANIFEST, "android").data)
    assert pins == {
        "release": RELEASE, "commit": COMMIT,
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/"
               f"llama-{RELEASE}-bin-linux-arm64-snapdragon.tar.gz",
        "size": 26027402, "sha256": "0fca14a6c6427f9baab799dc4324a06423fbc7de8944cc829eeb93d897924d1a",
        "files": {
            "v73": {"name": "libggml-htp-v73.so", "member": f"llama-{RELEASE}/lib/libggml-htp-v73.so", "size": 1009416,
                    "sha256": "7f1d429176aa3d2683df8d984f5645c818a6468aa3cc362c71420b71b60609ee"},
            "v75": {"name": "libggml-htp-v75.so", "member": f"llama-{RELEASE}/lib/libggml-htp-v75.so", "size": 1013512,
                    "sha256": "f977cf79cdb6edf564a74c5e6aabf707fb1ea871b707ab907a0e839d8a1a660e"},
            "v79": {"name": "libggml-htp-v79.so", "member": f"llama-{RELEASE}/lib/libggml-htp-v79.so", "size": 1034088,
                    "sha256": "d92f8f270d799184ba22893568fa574335e06e1bc3aa60384d27b1a8d554b01b"},
            "v81": {"name": "libggml-htp-v81.so", "member": f"llama-{RELEASE}/lib/libggml-htp-v81.so", "size": 1046408,
                    "sha256": "f635668d342ff7cb7c088795d549bc8a94ba11a4c8afa2384e610ad61c925526"},
            "v85": {"name": "libggml-htp-v85.so", "member": f"llama-{RELEASE}/lib/libggml-htp-v85.so", "size": 1046408,
                    "sha256": "35cb77a2447b98451b2c5919bdf1feba5d0b1b32e9786341c10cb1e27488201d"},
        }}
    # the pins stay readable (and checked) while the backend is switched off
    assert wh.hexagon_skeleton_pins(wh.cmw.tomllib.loads(_manifest_text(enabled=False))) == pins


def _manifest_text(enabled: bool) -> str:
    text = wh.DEFAULT_MANIFEST.read_text(encoding="utf-8").replace("\r\n", "\n")
    return text.replace("[hexagon]\nenabled = true\n", f"[hexagon]\nenabled = {'true' if enabled else 'false'}\n")


def test_skeleton_pins_of_the_toml_and_the_patched_loader_are_the_same():
    """wheelhouse.toml [hexagon] and mobile.patch's npu_skeleton_source() must name the same release, commit,
    archive and files; the commit must be the llama.cpp commit the recipe says 0.3.37 vendors."""
    toml = wh.hexagon_skeleton_pins(wh.Ctx(wh.DEFAULT_MANIFEST, "android").data)
    ns = _patched_cpu_loader(_real_ctypes_ns())
    assert ns["npu_skeleton_source"]() == toml
    # a copy: callers cannot change the constant
    copy = ns["npu_skeleton_source"]()
    copy["files"]["v79"]["size"] = 1
    assert ns["npu_skeleton_source"]()["files"]["v79"]["size"] == toml["files"]["v79"]["size"]
    # the recipe records the vendored llama.cpp commit (abbreviated) next to the version pin
    recipe = RECIPE.read_text(encoding="utf-8")
    vendored = re.findall(r"llama\.cpp ([0-9a-f]{8}) of\b", recipe)
    assert vendored and all(toml["commit"].startswith(v) for v in vendored), vendored
    assert re.search(r'^  version: "0\.3\.37"', recipe, re.M)
    # and the patch header names the same release and commit
    header = PATCH.read_text(encoding="utf-8").split("diff --git", 1)[0]
    assert f"release {toml['release']}" in header and toml["commit"][:8] in header


def test_switched_off_manifest_expects_and_fetches_nothing(tmp_path):
    ctx = wh.Ctx(_manifest(tmp_path, enabled=False), "android", wh.DEFAULT_PYPROJECT)
    assert wh.hexagon_spec(ctx.data) is None and wh.hexagon_targets(ctx) == {}
    assert wh.builds_hexagon_recipe(ctx, ["llama-cpp-python"])          # the recipe still loads the cache (OFF)
    assert not wh.needs_hexagon_sdk(ctx, ["llama-cpp-python"])
    plan = {Path(p).name for _u, p, _s in wh.fetch_plan(ctx, ["llama-cpp-python"], tmp_path / "dl")}
    assert not any(n.startswith("hexagon-sdk") for n in plan)
    assert not any("snapdragon" in n for n in plan)                    # the skeleton archive is never fetched in CI
    assert wh.profile_libs(wh.SITE_PROFILES["llama-cpp-python"], "arm64-v8a", wh.hexagon_targets(ctx)) == \
        wh.profile_libs(wh.SITE_PROFILES["llama-cpp-python"], "arm64-v8a")


def test_fetch_plan_puts_the_sdk_where_ci_caches_it(tmp_path):
    ctx = wh.Ctx(wh.DEFAULT_MANIFEST, "android")
    sdk = wh.hexagon_spec(ctx.data)["sdk"]
    plan = wh.fetch_plan(ctx, ["llama-cpp-python"], tmp_path / "dl", tmp_path / "cache")
    assert (sdk["url"], tmp_path / "cache" / sdk["file"], sdk["sha256"]) in plan
    assert not any("snapdragon" in str(p) for _u, p, _s in plan)
    plan = wh.fetch_plan(ctx, ["llama-cpp-python"], tmp_path / "dl")
    assert (sdk["url"], tmp_path / "dl" / sdk["file"], sdk["sha256"]) in plan
    assert not any(p.name == sdk["file"] for _u, p, _s in wh.fetch_plan(ctx, ["cryptography", "pillow"], tmp_path))


def test_env_exports_the_sdk_cache_key(tmp_path, capsys):
    args = types.SimpleNamespace(format="github", constraints_out=None)
    wh.cmd_env(wh.Ctx(wh.DEFAULT_MANIFEST, "android"), args)
    out = capsys.readouterr().out
    assert "hexagon_sdk_sha256=00725fc449916d7b9af2c2a3c6b54d61f3c872d5dbc1d3ca3fd2f839b1d5dc70" in out.splitlines()
    assert "hexagon_sdk_file=hexagon-sdk-v6.7.0.0-amd64-lnx.tar.xz" in out.splitlines()
    wh.cmd_env(wh.Ctx(_manifest(tmp_path, enabled=False), "android", wh.DEFAULT_PYPROJECT), args)
    lines = capsys.readouterr().out.splitlines()
    assert "hexagon_sdk_sha256=" in lines and "hexagon_sdk_file=" in lines
    wh.cmd_env(wh.Ctx(wh.DEFAULT_MANIFEST, "ios"), args)
    assert "hexagon_sdk" not in capsys.readouterr().out


_SKELETON_TABLE = (
    'skeleton_release = "b1"\nskeleton_commit = "' + "c" * 40 + '"\n'
    'skeleton_archive = { url = "https://github.com/ggml-org/llama.cpp/releases/download/b1/x.tar.gz", size = 9, '
    'sha256 = "' + "b" * 64 + '" }\n'
    '[hexagon.skeleton_files]\n'
    'v79 = { name = "libggml-htp-v79.so", member = "llama-b1/lib/libggml-htp-v79.so", size = 5, sha256 = "'
    + "d" * 64 + '" }\n')


@pytest.mark.parametrize("change, skeletons, message", [
    ({"enabled": '"yes"'}, None, "needs enabled = true or false"),
    ({"abis": '["arm64-v8a", "x86_64"]'}, None, "abis must be ['arm64-v8a']"),
    ({"htp_archs": '["73"]'}, None, "htp_archs must be distinct HTP arch names"),
    ({"htp_archs": "[]"}, None, "htp_archs must be distinct HTP arch names"),
    ({"tools": '"latest"'}, None, "tools must be a version string"),
    ({"sdk": '{ url = "http://x/y.tar.xz", sha256 = "' + "0" * 64 + '", file = "y.tar.xz" }'}, None, "an https url"),
    ({"sdk": '{ url = "https://x/y.tar.xz", file = "y.tar.xz" }'}, None, "needs sdk = {url, sha256, file}"),
    # the skeleton pins
    ({}, "", "needs skeleton_release"),
    ({}, _SKELETON_TABLE.replace('"b1"\n', '"latest"\n'), "needs skeleton_release"),
    ({}, _SKELETON_TABLE.replace("c" * 40, "c" * 7), "skeleton_commit (its 40-digit commit)"),
    ({}, _SKELETON_TABLE.replace("download/b1/", "download/b2/"), "skeleton_archive needs a url of llama.cpp's b1"),
    ({}, _SKELETON_TABLE.replace("size = 9, ", ""), "skeleton_archive needs"),
    ({"htp_archs": '["v79", "v81"]'}, _SKELETON_TABLE, "must name exactly the htp_archs ['v79', 'v81']"),
    ({}, _SKELETON_TABLE.replace('name = "libggml-htp-v79.so"', 'name = "x.so"'), "v79 needs name"),
    ({}, _SKELETON_TABLE.replace("/lib/libggml-htp-v79.so", "/bin/libggml-htp-v79.so"), "a member ending in"),
    ({}, _SKELETON_TABLE.replace("size = 5", "size = 0"), "v79 needs name"),
    ({}, _SKELETON_TABLE.replace("d" * 64, "D" * 64), "v79 needs name"),
])
def test_bad_hexagon_tables_are_refused(change, skeletons, message):
    table = {"enabled": "true", "abis": '["arm64-v8a"]', "version": '"6.7.0.0"', "tools": '"21.0.04"',
             "htp_archs": '["v79"]', "sdk": '{ url = "https://x/y.tar.xz", sha256 = "' + "a" * 64 + '", file = "y.tar.xz" }'}
    table.update(change)
    text = "[hexagon]\n" + "".join(f"{k} = {v}\n" for k, v in table.items()) + \
        (_SKELETON_TABLE if skeletons is None else skeletons)
    data = wh.cmw.tomllib.loads(text)
    with pytest.raises(wh.WheelsError, match=re.escape(message)):
        wh.hexagon_spec(data)
    assert wh.hexagon_spec({}) is None and wh.hexagon_spec({"hexagon": {"enabled": False}}) is None


def test_a_good_hexagon_table_parses():
    table = ('[hexagon]\nenabled = true\nabis = ["arm64-v8a"]\nversion = "6.7.0.0"\ntools = "21.0.04"\n'
             'htp_archs = ["v79"]\nsdk = { url = "https://x/y.tar.xz", sha256 = "' + "a" * 64 + '", file = "y.tar.xz" }\n'
             + _SKELETON_TABLE)
    spec = wh.hexagon_spec(wh.cmw.tomllib.loads(table))
    assert spec["skeletons"]["files"] == {"v79": {"name": "libggml-htp-v79.so", "member": "llama-b1/lib/libggml-htp-v79.so",
                                                  "size": 5, "sha256": "d" * 64}}


# ---- stage-hexagon with a tiny fake SDK ---------------------------------------------------------------------------

def _fake_sdk(path: Path, members=None, top="6.7.0.0", link=None) -> str:
    """A .tar.xz shaped like the Hexagon SDK tarball (top-level 6.7.0.0/); returns its sha256."""
    members = members if members is not None else {
        f"{top}/build/cmake/hexagon_fun.cmake": b"# fake\n",
        f"{top}/tools/HEXAGON_Tools/21.0.04/Tools/bin/hexagon-clang": b"#!/bin/sh\n",
        f"{top}/incs/remote.h": b"/* fake */\n"}
    with tarfile.open(path, "w:xz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
        if link:
            info = tarfile.TarInfo(link[0])
            info.type = tarfile.SYMTYPE
            info.linkname = link[1]
            tf.addfile(info)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage(ctx, tmp_path, packages=("llama-cpp-python",)):
    args = types.SimpleNamespace(packages=list(packages), downloads=str(tmp_path / "dl"), ndk=str(tmp_path / "ndk"),
                                 work=str(tmp_path / "work"))
    wh.cmd_stage_hexagon(ctx, args)
    return tmp_path / "ndk" / wh.HEXAGON_INIT_CACHE


def test_stage_hexagon_unpacks_the_sdk_once_and_writes_the_initial_cache(tmp_path):
    (tmp_path / "dl").mkdir()
    sha = _fake_sdk(tmp_path / "dl" / "hexagon-sdk-v6.7.0.0-amd64-lnx.tar.xz",
                    link=("6.7.0.0/tools/HEXAGON_Tools/21.0.04/Tools/bin/clang", "hexagon-clang"))
    ctx = wh.Ctx(_manifest(tmp_path, sdk_sha=sha), "android", wh.DEFAULT_PYPROJECT)
    init = _stage(ctx, tmp_path)
    root = tmp_path / "work" / "hexagon-sdk-6.7.0.0" / "6.7.0.0"
    text = init.read_text(encoding="utf-8")
    assert 'set(GGML_HEXAGON ON CACHE BOOL "" FORCE)' in text
    assert f'set(HEXAGON_SDK_ROOT "{root.as_posix()}" CACHE PATH "" FORCE)' in text
    assert f'set(HEXAGON_TOOLS_ROOT "{(root / "tools" / "HEXAGON_Tools" / "21.0.04").as_posix()}"' in text
    assert 'set(PREBUILT_LIB_DIR "android_aarch64" CACHE STRING "" FORCE)' in text
    # the second run reuses the unpacked tree (the marker holds the tarball's sha256)
    sentinel = root / "kept.txt"
    sentinel.write_text("x", encoding="utf-8")
    _stage(ctx, tmp_path)
    assert sentinel.is_file()
    # nothing to stage without the Android arm64 engine
    (tmp_path / "ndk2").mkdir()
    wh.cmd_stage_hexagon(ctx, types.SimpleNamespace(packages=["pillow"], downloads="x", ndk=str(tmp_path / "ndk2"),
                                                    work="x"))
    wh.cmd_stage_hexagon(wh.Ctx(_manifest(tmp_path, sdk_sha=sha), "ios", wh.DEFAULT_PYPROJECT),
                         types.SimpleNamespace(packages=["llama-cpp-python"], downloads="x", ndk=str(tmp_path / "ndk2"),
                                               work="x"))
    assert not (tmp_path / "ndk2" / wh.HEXAGON_INIT_CACHE).exists()


def test_stage_hexagon_switched_off_writes_off_and_downloads_nothing(tmp_path):
    init = _stage(wh.Ctx(_manifest(tmp_path, enabled=False), "android", wh.DEFAULT_PYPROJECT), tmp_path)
    assert init.read_text(encoding="utf-8").splitlines()[1:] == ['set(GGML_HEXAGON OFF CACHE BOOL "" FORCE)']
    assert not (tmp_path / "work").exists()


def test_stage_hexagon_command_line(tmp_path, capsys):
    manifest = _manifest(tmp_path, enabled=False)
    argv = ["--manifest", str(manifest), "stage-hexagon", "--platform", "android", "--downloads", str(tmp_path / "dl"),
            "--ndk", str(tmp_path / "ndk"), "--work", str(tmp_path / "work"), "llama-cpp-python"]
    assert wh.main(argv) == 0
    assert "GGML_HEXAGON OFF" in (tmp_path / "ndk" / wh.HEXAGON_INIT_CACHE).read_text(encoding="utf-8")
    # enabled without the fetched tarball: a clean error, exit 1
    argv[1] = str(_manifest(tmp_path / "on"))
    assert wh.main(argv) == 1
    assert "missing or its sha256 differs from the manifest (run fetch)" in capsys.readouterr().err


def test_stage_hexagon_refuses_a_wrong_or_broken_sdk(tmp_path):
    (tmp_path / "dl").mkdir()
    tarball = tmp_path / "dl" / "hexagon-sdk-v6.7.0.0-amd64-lnx.tar.xz"
    _fake_sdk(tarball)
    with pytest.raises(wh.WheelsError, match="missing or its sha256 differs"):
        _stage(wh.Ctx(wh.DEFAULT_MANIFEST, "android"), tmp_path)          # the real pin: not this fake
    sha = _fake_sdk(tarball, members={"6.7.0.0/incs/remote.h": b"x"})
    with pytest.raises(wh.WheelsError, match="no 6.7.0.0/build/cmake/hexagon_fun.cmake after extraction"):
        _stage(wh.Ctx(_manifest(tmp_path, sdk_sha=sha), "android", wh.DEFAULT_PYPROJECT), tmp_path)
    sha = _fake_sdk(tarball, members={"6.7.0.0/build/cmake/hexagon_fun.cmake": b"x"})
    with pytest.raises(wh.WheelsError, match="tools/HEXAGON_Tools/21.0.04"):
        _stage(wh.Ctx(_manifest(tmp_path, sdk_sha=sha), "android", wh.DEFAULT_PYPROJECT), tmp_path)
    sha = _fake_sdk(tarball, members={"../evil": b"x"})
    with pytest.raises(wh.WheelsError, match="unsafe member"):
        _stage(wh.Ctx(_manifest(tmp_path, sdk_sha=sha), "android", wh.DEFAULT_PYPROJECT), tmp_path)
    # a link out of the tree: tarfile's data filter refuses it
    sha = _fake_sdk(tarball, link=("6.7.0.0/incs/out", "/etc/passwd"))
    with pytest.raises(wh.WheelsError, match="LinkError"):
        _stage(wh.Ctx(_manifest(tmp_path, sdk_sha=sha), "android", wh.DEFAULT_PYPROJECT), tmp_path)


def test_recipe_loads_the_hexagon_initial_cache_on_android_arm64_only():
    for sdk, arch in (("android", "arm64-v8a"), ("android", "x86_64"), ("iphoneos", "arm64"),
                      ("iphonesimulator", "arm64"), ("iphonesimulator", "x86_64")):
        args = _render_recipe(RECIPE, sdk, arch)["build"]["script_env"]["CMAKE_ARGS"].split()
        caches = [a for a in args if a.startswith("-C")]
        if (sdk, arch) == ("android", "arm64-v8a"):
            assert caches == ["-C{NDK_ROOT}/" + wh.HEXAGON_INIT_CACHE], args
        else:
            assert caches == [], (sdk, arch, args)
        # the switch lives in the initial cache only: no -DGGML_HEXAGON / SDK path in the recipe
        assert not any("HEXAGON" in a for a in args), args


# ---- verify-native / verify-site-packages ---------------------------------------------------------------------------

def make_elf32(machine: int = wh.EM_QDSP6, big: bool = False, extra: bytes = b"") -> bytes:
    """A 32-bit ELF header (a Hexagon DSP shared object: ELFCLASS32, EM_QDSP6 by default)."""
    end = ">" if big else "<"
    ident = b"\x7fELF" + bytes([1, 2 if big else 1, 1, 0]) + b"\0" * 8
    return ident + struct.pack(end + "HHIIIIIHHHHHH", 3, machine, 1, 0, 52, 0, 0, 52, 32, 0, 40, 0, 0) + extra


def test_elf_machine_reads_both_classes_and_byte_orders():
    assert wh.elf_machine(make_elf32()) == 164 and wh.elf_machine(make_elf32(big=True)) == 164
    assert wh.elf_machine(make_elf()) == 183
    with pytest.raises(ValueError):
        wh.elf_machine(b"MZ" + b"\0" * 40)


def make_npu_android(sp: Path, libs=HEX_LIBS_ARM64, marker=True, extra_files=None, **kw) -> Path:
    """build/site-packages/arm64-v8a with the engine, libggml-hexagon and its marker (``extra_files``:
    {relative path: bytes} written on top, e.g. a skeleton that must not ship)."""
    arch = make_llama_android(sp, "arm64-v8a", libs=libs, **kw)
    if marker:
        (arch / wh.NPU_MARKER).write_text("libggml-hexagon.so\n", encoding="utf-8")
    for rel, data in (extra_files or {}).items():
        (arch / rel).parent.mkdir(parents=True, exist_ok=True)
        (arch / rel).write_bytes(data)
    return arch


HEXAGON = {"arm64-v8a": ARCHS}


def test_verify_site_packages_accepts_the_npu_backend(tmp_path):
    sp = tmp_path / "sp"
    make_npu_android(sp)
    errors, notes = _verify_llama(sp, "android", make_ndk(tmp_path / "ndk", LLAMA_NDK_EXPORTS), hexagon=HEXAGON)
    assert errors == []
    assert "arm64-v8a: llama_cpp/npu-backend.txt marks the NPU backend; no HTP skeleton ships" in notes
    assert any("lib/libggml-hexagon.so NEEDED" in n for n in notes)
    assert "arm64-v8a: lib/libggml-hexagon.so armv8.0 only: optional instructions none" in notes


@pytest.mark.parametrize("extra", [
    {"llama_cpp/htp/libggml-htp-v79.so.dsp": make_elf32()},          # the old package data
    {"llama_cpp/lib/libggml-htp-v79.so": make_elf32()},              # upstream's install(... TYPE LIB)
    {"llama_cpp/lib/libggml-htp-v81.so": b"not even an ELF"},        # by name alone
    {"other_pkg/data/v73.so.dsp": b"x"},                             # any *.so.dsp, in any package
    {"other_pkg/data/blob.bin": make_elf32()},                       # a Hexagon ELF under any name
])
def test_verify_site_packages_fails_on_any_shipped_skeleton(tmp_path, extra):
    sp = tmp_path / "sp"
    make_npu_android(sp, extra_files=extra)
    ndk = make_ndk(tmp_path / "ndk", LLAMA_NDK_EXPORTS)
    for hexagon in (HEXAGON, {}):                                      # on or off: never a skeleton
        errors, _ = _verify_llama(sp, "android", ndk, hexagon=hexagon)
        assert any(f"HTP skeleton files ['{next(iter(extra))}']: they may not ship" in e for e in errors), errors


@pytest.mark.parametrize("kwargs, hexagon, message", [
    # the backend library missing / its marker missing
    ({"libs": LLAMA_LIBS_ARM64}, HEXAGON, "expected exactly ['libggml', 'libggml-base', 'libggml-cpu-android_armv8.0_1'"),
    ({"marker": False}, HEXAGON, "arm64-v8a: no llama_cpp/npu-backend.txt (the NPU backend's marker"),
    # [hexagon] switched off, but the build still carries it
    ({}, None, "llama_cpp/npu-backend.txt is there, but no NPU backend is expected ([hexagon] is switched off)"),
    ({}, None, "holds ['libggml', 'libggml-base', 'libggml-cpu-android_armv8.0_1', 'libggml-cpu-android_armv8.2_1', "
               "'libggml-cpu-android_armv8.2_2', 'libggml-cpu-android_armv8.6_1', 'libggml-hexagon'"),
    # nothing may NEED the NPU backend; it may not NEED the vendor's FastRPC library
    ({"extra_needed": ("libggml-hexagon.so",)}, HEXAGON, "only loaded on demand, never NEEDED"),
    ({"extra_needed": ("libcdsprpc.so",)}, HEXAGON, "NEEDED ['libcdsprpc.so'] beyond"),
])
def test_verify_site_packages_rejects_npu_mistakes(tmp_path, kwargs, hexagon, message):
    sp = tmp_path / "sp"
    make_npu_android(sp, **kwargs)
    errors, _ = _verify_llama(sp, "android", make_ndk(tmp_path / "ndk", LLAMA_NDK_EXPORTS), hexagon=hexagon)
    assert any(message in e for e in errors), errors


def test_npu_skeleton_files_names_and_elf_machines():
    heads = {"a/libggml-htp-v79.so": None, "b/x.so.dsp": None, "c/blob": make_elf32()[:20], "d/arm.so": make_elf()[:20],
             "e/text.txt": b"hello", "f/libggml-hexagon.so": make_elf()[:20], "g/short": b"\x7fEL"}
    assert wh.npu_skeleton_files(list(heads), heads.get) == ["a/libggml-htp-v79.so", "b/x.so.dsp", "c/blob"]


def test_switched_off_build_without_npu_files_passes(tmp_path):
    sp = tmp_path / "sp"
    make_llama_android(sp, "arm64-v8a")
    errors, _ = _verify_llama(sp, "android", make_ndk(tmp_path / "ndk", LLAMA_NDK_EXPORTS), hexagon={})
    assert errors == []
    # ... but with [hexagon] on, the same build is missing the backend and its marker
    errors, _ = _verify_llama(sp, "android", make_ndk(tmp_path / "ndk2", LLAMA_NDK_EXPORTS), hexagon=HEXAGON)
    text = "\n".join(errors)
    assert "no llama_cpp/npu-backend.txt" in text and "libggml-hexagon" in text


def test_verify_native_checks_the_npu_files_of_the_wheel(tmp_path):
    ndk = make_ndk(tmp_path / "ndk", LLAMA_NDK_EXPORTS)
    d = ndk / "toolchains" / "llvm" / "prebuilt" / "linux-x86_64" / "sysroot" / "usr" / "lib" / "aarch64-linux-android"
    (d / "libc++_shared.so").write_bytes(make_elf(needed=("libc.so",), symbols=[("_ZNSt6__ndk14cout", base.GLOBAL,
                                                                                   base.DEF)]))
    stubs = wh.NdkStubs(ndk)
    wheel = _zip_tree(make_npu_android(tmp_path / "sp"), tmp_path / "llama_cpp_python-0.3.37-1000-cp313-cp313-a.whl")
    errors, notes = wh.verify_native_wheel(wheel, "llama-cpp-python", "arm64-v8a", "3.5.9", (13, 0), stubs, HEXAGON)
    assert errors == [], errors
    assert any("llama_cpp/npu-backend.txt marks the NPU backend; no HTP skeleton ships" in n for n in notes)
    # the switched-off expectation refuses the same wheel; a wheel without the backend fails the enabled one
    errors, _ = wh.verify_native_wheel(wheel, "llama-cpp-python", "arm64-v8a", "3.5.9", (13, 0), stubs, {})
    assert any("([hexagon] is switched off)" in e for e in errors), errors
    bare = _zip_tree(make_llama_android(tmp_path / "sp2", "arm64-v8a"), tmp_path / "bare.whl")
    errors, _ = wh.verify_native_wheel(bare, "llama-cpp-python", "arm64-v8a", "3.5.9", (13, 0), stubs, HEXAGON)
    assert any("no llama_cpp/npu-backend.txt" in e for e in errors), errors
    # a wheel that still carries the old package-data skeletons, or upstream's installed ones, fails
    for i, rel in enumerate(("llama_cpp/htp/libggml-htp-v79.so.dsp", "llama_cpp/lib/libggml-htp-v79.so")):
        old = _zip_tree(make_npu_android(tmp_path / f"old{i}", extra_files={rel: make_elf32()}), tmp_path / f"old{i}.whl")
        errors, _ = wh.verify_native_wheel(old, "llama-cpp-python", "arm64-v8a", "3.5.9", (13, 0), stubs, HEXAGON)
        assert any(f"HTP skeleton files ['{rel}']" in e for e in errors), errors
    # x86_64 never carries it, whatever the manifest says about arm64
    x86 = _zip_tree(make_llama_android(tmp_path / "sp3", "x86_64"), tmp_path / "x86.whl")
    (d.parent / "x86_64-linux-android").mkdir(exist_ok=True)
    (d.parent / "x86_64-linux-android" / "libc++_shared.so").write_bytes(make_elf(
        machine=62, needed=("libc.so",), symbols=[("_ZNSt6__ndk14cout", base.GLOBAL, base.DEF)]))
    errors, _ = wh.verify_native_wheel(x86, "llama-cpp-python", "x86_64", "3.5.9", (13, 0), stubs, HEXAGON)
    assert errors == [], errors


# ---- the patched loader: npu_precheck / register_npu_backend / device_index over a fake ctypes --------------------

ARCH_V79 = 0x79  # DSPRPC_GET_DSP_INFO ARCH_VER reports the arch as BCD-like nibbles (htp-drv.cpp)
V79_SIZE = 1034088


def _npu_ctypes(tmp_path, arch_value=ARCH_V79, rpc=True, rpc_control=True, load_ok=True, session=0,
                session_symbol=True, devices=("HTP0",), unsigned=1, control_err=0):
    """A fake ctypes (real Structure/byref/buffers, fake CDLL) + libggml for register_npu_backend."""
    calls = {"cdll": [], "load": [], "env_at_load": [], "session": []}

    def remote_handle_control(req, cap_ref, size):
        cap = cap_ref._obj
        calls.setdefault("control", []).append((req, cap.domain, cap.attribute_ID, size))
        if control_err:
            return control_err
        cap.capability = arch_value if cap.attribute_ID == 6 else unsigned
        return 0

    def session_check(index, err, size):
        calls["session"].append(index)
        if session:
            err.value = b"ggml-hex: remote_session_control(unsign) failed (see log for details)"
        return session

    def cdll(name, mode=0):
        calls["cdll"].append(name)
        if name == "libcdsprpc.so":
            if not rpc:
                raise OSError('dlopen failed: library "libcdsprpc.so" not found')
            return types.SimpleNamespace(**({"remote_handle_control": _FakeFn(remote_handle_control)}
                                            if rpc_control else {}))
        if name.endswith("libggml-hexagon.so"):
            return types.SimpleNamespace(**({"glossarion_hexagon_session_check": _FakeFn(session_check)}
                                            if session_symbol else {}))
        raise OSError(f"dlopen failed: {name}")

    def load(path):
        calls["load"].append(path.decode())
        calls["env_at_load"].append({k: os.environ.get(k) for k in
                                     ("GGML_HEXAGON_LIBHTP_DIR", "GGML_HEXAGON_ARCH", "GGML_HEXAGON_POWER_MODE")})
        return 0x48 if load_ok else None

    libggml = types.SimpleNamespace(
        ggml_backend_load=_FakeFn(load),
        ggml_backend_reg_dev_count=_FakeFn(lambda reg: len(devices)),
        ggml_backend_reg_dev_get=_FakeFn(lambda reg, i: 100 + i),
        ggml_backend_dev_name=_FakeFn(lambda dev: devices[dev - 100].encode()))
    names = {k: getattr(real_ctypes, k) for k in dir(real_ctypes) if not k.startswith("__")}
    ns = types.SimpleNamespace(**{**names, "CDLL": cdll})
    return ns, libggml, calls


ENV_KEYS = ("GGML_HEXAGON_LIBHTP_DIR", "GGML_HEXAGON_ARCH", "GGML_HEXAGON_POWER_MODE")


def _loader(tmp_path, monkeypatch, marker=True, **kw):
    for key in ENV_KEYS:
        monkeypatch.setenv(key, "")          # restored after the test; the loader sets them
    ns_ctypes, libggml, calls = _npu_ctypes(tmp_path, **kw)
    ns = _patched_cpu_loader(ns_ctypes)
    pkg = tmp_path / "pkg" / "llama_cpp"
    lib = pkg / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    (lib / "libggml-hexagon.so").write_bytes(b"")
    if marker:
        (pkg / "npu-backend.txt").write_text("libggml-hexagon.so\n", encoding="utf-8")
    ns["_NPU_MARKER_SOURCE"], ns["_LIBGGML"], ns["_BASE_PATH"] = pkg / "npu-backend.txt", libggml, lib
    return ns, calls


def _download(skel: Path, arch="v79", size=V79_SIZE) -> Path:
    """The app's download: libggml-htp-<arch>.so of the pinned size in skel_dir (contents do not matter here)."""
    skel.mkdir(parents=True, exist_ok=True)
    path = skel / f"libggml-htp-{arch}.so"
    with open(path, "wb") as stream:
        stream.truncate(size)
    return path


def _steps(state):
    return [(s["name"], s["ok"]) for s in state["steps"]]


def test_register_npu_backend_sets_the_environment_before_the_only_load(tmp_path, monkeypatch):
    ns, calls = _loader(tmp_path, monkeypatch)
    assert ns["NPU_STATE"] == {} and calls["load"] == []                        # nothing loads on import
    assert ns["npu_backend_shipped"]() is True and calls["cdll"] == []           # naming it loads nothing
    skel = tmp_path / "files" / "htp"
    _download(skel)
    state = ns["register_npu_backend"](str(skel), {"GGML_HEXAGON_POWER_MODE": "balanced", "PATH": "/x",
                                                   "GGML_HEXAGON_LIBHTP_DIR": "/elsewhere"})
    assert state["backend"] == "HTP" and state["device"] == "HTP0" and state["arch"] == 79 and state["error"] == ""
    assert _steps(state) == [(n, True) for n in ("shipped", "libcdsprpc", "arch", "skeletons", "registry", "session")]
    assert "Hexagon v79; unsigned PD supported" in state["steps"][2]["detail"]
    assert state["steps"][3]["detail"] == f"libggml-htp-v79.so in {skel} (llama.cpp {RELEASE})"
    assert "ignored options ['PATH', 'GGML_HEXAGON_LIBHTP_DIR']" in state["steps"][4]["detail"]
    # our own DSPRPC_GET_DSP_INFO queries (cDSP domain 3): ARCH_VER (6), then UNSIGNED_PD_SUPPORT (1)
    assert calls["control"] == [(2, 3, 6, 12), (2, 3, 1, 12)]
    # nothing is copied or written: the directory holds only the download
    assert sorted(p.name for p in skel.iterdir()) == ["libggml-htp-v79.so"]
    # loaded once, by the file next to libggml, with the environment already set
    assert calls["load"] == [str(tmp_path / "pkg" / "llama_cpp" / "lib" / "libggml-hexagon.so")]
    assert calls["env_at_load"] == [{"GGML_HEXAGON_LIBHTP_DIR": str(skel), "GGML_HEXAGON_ARCH": "79",
                                     "GGML_HEXAGON_POWER_MODE": "balanced"}]
    assert calls["session"] == [0]
    # after the load: the first answer, no second load
    again = ns["register_npu_backend"](str(tmp_path / "other"), {})
    assert again == state and len(calls["load"]) == 1 and not (tmp_path / "other").exists()
    again["steps"][0]["ok"] = False
    assert ns["register_npu_backend"](str(skel))["steps"][0]["ok"] is True      # callers get copies


def test_register_npu_backend_without_the_download_registers_nothing_and_can_run_again(tmp_path, monkeypatch):
    ns, calls = _loader(tmp_path, monkeypatch)
    skel = tmp_path / "files" / "htp"
    for prepare in (lambda: None,                                                # no directory at all
                    lambda: _download(skel, arch="v81", size=1046408),           # another arch's file only
                    lambda: _download(skel, size=V79_SIZE - 1),                  # a partial download
                    lambda: (skel / "libggml-htp-v79.so").unlink() or (skel / "libggml-htp-v79.so").mkdir()):
        prepare()
        state = ns["register_npu_backend"](str(skel))
        assert _steps(state)[:4] == [("shipped", True), ("libcdsprpc", True), ("arch", True), ("skeletons", False)]
        assert state["steps"][3]["detail"] == "NPU files not downloaded" and state["error"] == "NPU files not downloaded"
        assert all(s["detail"] == "not checked (an earlier step failed)" for s in state["steps"][4:])
        assert state["arch"] == 79 and ns["NPU_STATE"] == {} and calls["load"] == []
        assert all(os.environ.get(k) == "" for k in ENV_KEYS)                    # nothing set for ggml-hexagon
    (skel / "libggml-htp-v79.so").rmdir()
    _download(skel)                                                               # the download finished
    state = ns["register_npu_backend"](str(skel))
    assert all(ok for _n, ok in _steps(state)) and len(calls["load"]) == 1 and ns["NPU_STATE"]["device"] == "HTP0"


def test_a_symlinked_skeleton_does_not_count(tmp_path, monkeypatch):
    ns, calls = _loader(tmp_path, monkeypatch)
    real = _download(tmp_path / "elsewhere")
    skel = tmp_path / "files" / "htp"
    skel.mkdir(parents=True)
    try:
        (skel / "libggml-htp-v79.so").symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("no symlinks here")
    state = ns["register_npu_backend"](str(skel))
    assert state["steps"][3] == {"name": "skeletons", "ok": False, "detail": "NPU files not downloaded"}
    assert calls["load"] == []


@pytest.mark.parametrize("kw, failed, detail", [
    ({"rpc": False}, "libcdsprpc", 'no FastRPC driver (dlopen failed: library "libcdsprpc.so" not found)'),
    ({"rpc_control": False}, "arch", "libcdsprpc.so has no remote_handle_control"),
    ({"control_err": -2147482624}, "arch", "remote_handle_control(DSPRPC_GET_DSP_INFO) failed (error 0x80000400)"),
    ({"arch_value": 0}, "arch", "the cDSP reported arch value 0x0"),
    ({"arch_value": 0x68}, "arch", f"Hexagon v68: llama.cpp {RELEASE} has NPU files for v73, v75, v79, v81, v85 only"),
    ({"load_ok": False}, "registry", "the Hexagon backend did not start (ggml_backend_init returned NULL"),
    ({"devices": ()}, "registry", "HTP registered without a device"),
    ({"session": -1}, "session", "ggml-hex: remote_session_control(unsign) failed (see log for details)"),
    ({"session_symbol": False}, "session", "not checked: this build has no session check"),
])
def test_register_npu_backend_reports_the_first_failure_and_never_raises(tmp_path, monkeypatch, kw, failed, detail):
    ns, calls = _loader(tmp_path, monkeypatch, **kw)
    skel = tmp_path / "files" / "htp"
    _download(skel)
    state = ns["register_npu_backend"](str(skel))
    names = [s["name"] for s in state["steps"]]
    assert names == ["shipped", "libcdsprpc", "arch", "skeletons", "registry", "session"]
    first = next(s for s in state["steps"] if not s["ok"])
    assert first["name"] == failed and detail in first["detail"], state
    assert state["error"] == first["detail"]
    after = state["steps"][names.index(failed) + 1:]
    assert all(not s["ok"] and s["detail"] == "not checked (an earlier step failed)" for s in after)
    if names.index(failed) < names.index("registry"):
        assert calls["load"] == [] and ns["NPU_STATE"] == {}        # nothing loads or stays after an early failure
    else:
        assert len(calls["load"]) == 1 and ns["NPU_STATE"] == state  # after the load the answer stays


def test_register_npu_backend_without_the_backend_or_the_libraries(tmp_path, monkeypatch):
    ns, calls = _loader(tmp_path, monkeypatch, marker=False)
    assert ns["npu_backend_shipped"]() is False
    state = ns["register_npu_backend"](str(tmp_path / "s"))
    assert _steps(state)[0] == ("shipped", False) and "carries no NPU backend" in state["error"]
    assert calls["cdll"] == []
    # the engine's libraries not loaded yet: the registry step says so, nothing is set, loaded or remembered
    ns, calls = _loader(tmp_path / "b", monkeypatch)
    ns["_LIBGGML"] = None
    _download(tmp_path / "b" / "s")
    state = ns["register_npu_backend"](str(tmp_path / "b" / "s"))
    assert ("registry", False) in _steps(state) and state["error"] == "the engine's libraries are not loaded"
    assert ns["NPU_STATE"] == {} and calls["load"] == [] and os.environ.get("GGML_HEXAGON_ARCH") == ""
    # an exception inside: reported on the first step not run, never raised
    ns, _calls = _loader(tmp_path / "c", monkeypatch)
    ns["_NPU_SKELETONS"] = {"release": RELEASE, "files": {"v79": {}}}
    state = ns["register_npu_backend"](str(tmp_path / "c" / "s"))
    bad = next(s for s in state["steps"] if not s["ok"])
    assert bad["name"] == "skeletons" and bad["detail"].startswith("KeyError")


def test_npu_backend_shipped_on_android_arm64_without_a_file(tmp_path, monkeypatch):
    ns, _calls = _loader(tmp_path, monkeypatch)
    ns["_BASE_PATH"] = tmp_path / "sitepackages.zip" / "llama_cpp" / "lib"      # jniLibs: no file on disk
    monkeypatch.setattr(ns["sys"], "platform", "android")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    assert ns["npu_backend_shipped"]() is True
    ns["_NPU_MARKER_SOURCE"] = tmp_path / "none.txt"                             # a build without the backend
    assert ns["npu_backend_shipped"]() is False
    ns["_NPU_MARKER_SOURCE"] = tmp_path / "pkg" / "llama_cpp" / "npu-backend.txt"
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    assert ns["npu_backend_shipped"]() is False
    state = ns["register_npu_backend"](str(tmp_path / "s"))
    assert _steps(state)[0] == ("shipped", False)


def test_npu_backend_shipped_reads_the_marker_inside_a_zip(tmp_path, monkeypatch):
    ns, _calls = _loader(tmp_path, monkeypatch)
    archive = tmp_path / "sitepackages.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("llama_cpp/npu-backend.txt", "libggml-hexagon.so\n")
    ns["_NPU_MARKER_SOURCE"] = zipfile.Path(archive, "llama_cpp/npu-backend.txt")
    assert ns["npu_backend_shipped"]() is True
    ns["_NPU_MARKER_SOURCE"] = zipfile.Path(archive, "llama_cpp/other.txt")
    assert ns["npu_backend_shipped"]() is False
    ns["_NPU_MARKER_SOURCE"] = None                                              # no llama_cpp package here
    assert ns["npu_backend_shipped"]() is False


@pytest.mark.parametrize("kw, steps, arch, error", [
    ({}, [("shipped", True), ("libcdsprpc", True), ("arch", True)], "v79", None),
    ({"rpc": False}, [("shipped", True), ("libcdsprpc", False), ("arch", False)], None,
     'no FastRPC driver (dlopen failed: library "libcdsprpc.so" not found)'),
    ({"arch_value": 0x68}, [("shipped", True), ("libcdsprpc", True), ("arch", False)], None,
     f"Hexagon v68: llama.cpp {RELEASE} has NPU files for v73, v75, v79, v81, v85 only"),
    ({"arch_value": 0x81}, [("shipped", True), ("libcdsprpc", True), ("arch", True)], "v81", None),
    ({"marker": False}, [("shipped", False), ("libcdsprpc", False), ("arch", False)], None,
     "this build carries no NPU backend (libggml-hexagon)"),
])
def test_npu_precheck_runs_the_cheap_steps_only(tmp_path, monkeypatch, kw, steps, arch, error):
    ns, calls = _loader(tmp_path, monkeypatch, **kw)
    for _ in range(2):                                                            # any number of calls
        result = ns["npu_precheck"]()
        assert result["arch"] == arch and result["error"] == error and _steps(result) == steps, result
        assert set(result) == {"arch", "error", "steps"}
    assert ns["NPU_PRECHECK_STEPS"] == ("shipped", "libcdsprpc", "arch")
    # no backend load, no hexagon dlopen, no environment, no state
    assert calls["load"] == [] and not any(n.endswith("libggml-hexagon.so") for n in calls["cdll"])
    assert all(os.environ.get(k) == "" for k in ENV_KEYS) and ns["NPU_STATE"] == {}


def test_npu_precheck_never_raises(tmp_path, monkeypatch):
    ns, _calls = _loader(tmp_path, monkeypatch)
    ns["_dsp_info"] = lambda control, attribute: 1 / 0
    result = ns["npu_precheck"]()
    assert _steps(result) == [("shipped", True), ("libcdsprpc", True), ("arch", False)]
    assert result["error"].startswith("ZeroDivisionError") and result["arch"] is None


# device_index: ggml's devices in registration order -> llama.cpp's model->devices order
CPU, GPU, IGPU, ACCEL = 0, 1, 2, 3


def _device_libggml(devices):
    """devices: [(registry name, type, device_id)] in ggml's registration order."""
    regs = {name: 1000 + i for i, name in enumerate(dict.fromkeys(d[0] for d in devices))}

    def props(dev, ref):
        ref._obj.device_id = devices[dev][2]
        ref._obj.type = devices[dev][1]

    return types.SimpleNamespace(
        ggml_backend_dev_count=_FakeFn(lambda: len(devices)),
        ggml_backend_dev_get=_FakeFn(lambda i: i),
        ggml_backend_dev_type=_FakeFn(lambda dev: devices[dev][1]),
        ggml_backend_dev_backend_reg=_FakeFn(lambda dev: regs[devices[dev][0]]),
        ggml_backend_reg_name=_FakeFn(lambda reg: next(n for n, r in regs.items() if r == reg).encode()),
        ggml_backend_dev_get_props=_FakeFn(props))


@pytest.mark.parametrize("devices, expected", [
    # the app's order: CPU variant, then OpenCL, then the NPU
    ([("CPU", CPU, None), ("OpenCL", GPU, None), ("HTP", GPU, None)], {"OpenCL": 0, "HTP": 1, "Metal": -1}),
    # the NPU registered first
    ([("CPU", CPU, None), ("HTP", GPU, None), ("OpenCL", GPU, None)], {"HTP": 0, "OpenCL": 1}),
    # RPC servers go first; ACCEL devices are skipped like the CPU
    ([("BLAS", ACCEL, None), ("HTP", GPU, None), ("RPC", GPU, None)], {"RPC": 0, "HTP": 1, "BLAS": -1}),
    # integrated GPUs only when there is no GPU
    ([("CPU", CPU, None), ("Vulkan", IGPU, None), ("HTP", GPU, None)], {"HTP": 0, "Vulkan": -1}),
    ([("CPU", CPU, None), ("Vulkan", IGPU, None)], {"Vulkan": 0}),
    # a second GPU with the same device id is skipped (llama.cpp compares ids only when both are set)
    ([("CUDA", GPU, b"0000:01:00.0"), ("Vulkan", GPU, b"0000:01:00.0"), ("HTP", GPU, None)],
     {"CUDA": 0, "Vulkan": -1, "HTP": 1}),
])
def test_device_index_follows_llama_cpp_device_order(devices, expected):
    ns = _patched_cpu_loader(_real_ctypes_ns())
    assert ns["device_index"]("HTP") == -1                                   # before the libraries load
    ns["_LIBGGML"] = _device_libggml(devices)
    for name, index in expected.items():
        assert ns["device_index"](name) == index, (name, devices)


def test_device_index_quotes_the_llama_cpp_lines_it_mirrors():
    upstream = ("model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());",
                "model->devices.insert(model->devices.end(), gpus.begin(), gpus.end());",
                "if (igpus.empty() || ggml_backend_dev_backend_reg(dev) == ggml_backend_dev_backend_reg(igpus.back().dev)) {",
                "if (ggml_backend_reg_name(reg) == std::string(\"RPC\")) {")
    added = "\n".join(line[1:] for line in PATCH.read_text(encoding="utf-8").splitlines() if line.startswith("+"))
    for line in upstream:
        assert line in added, line


# ---- the patch itself ---------------------------------------------------------------------------------------------

def _file_diffs(text: str) -> dict:
    return {m.group(1): m.group(2) for m in re.finditer(r"^diff --git a/(\S+) b/\S+\n(.*?)(?=^diff --git |\Z)", text,
                                                        re.M | re.S)}


def test_patch_hunk_headers_match_their_bodies():
    """Every hunk's @@ -a,b +c,d @@ counts its body exactly, and the new starts follow from the earlier hunks of
    the same file (the patch is edited by hand; a wrong count only shows when forge applies it in CI)."""
    text = PATCH.read_text(encoding="utf-8")
    assert "\r" not in text
    total = 0
    for name, body in _file_diffs(text).items():
        lines = body.split("\n")
        assert lines[1] == f"--- a/{name}" and lines[2] == f"+++ b/{name}", name
        shift, last_old_end, i = 0, 0, 3
        while i < len(lines) and lines[i].startswith("@@ "):
            m = re.match(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@", lines[i])
            assert m, lines[i]
            a, b, c, d = map(int, m.groups())
            i += 1
            old = new = 0
            while i < len(lines) and lines[i][:1] in (" ", "-", "+") and not lines[i].startswith("@@ "):
                old += lines[i][0] in " -"
                new += lines[i][0] in " +"
                i += 1
            assert (old, new) == (b, d), f"{name} @@ -{a},{b} +{c},{d} @@: the body has -{old} +{new}"
            assert a > last_old_end and c == a + shift, f"{name} @@ -{a},{b} +{c},{d} @@: new start should be {a + shift}"
            shift += d - b
            last_old_end = a + b - 1
            total += 1
        assert all(line == "" for line in lines[i:]), f"{name}: unexpected text after the hunks: {lines[i:i + 2]}"
    assert total >= 16


def test_patch_builds_and_installs_no_skeleton_and_adds_the_session_check():
    text = PATCH.read_text(encoding="utf-8")
    assert "10. Optional NPU backend (Qualcomm Hexagon HTP, ggml-hexagon)" in text.split("diff --git", 1)[0]
    hunks = _file_diffs(text)
    cmake = hunks["vendor/llama.cpp/ggml/src/ggml-hexagon/CMakeLists.txt"]
    removed = [line[1:] for line in cmake.splitlines() if line.startswith("-") and not line.startswith("---")]
    added = [line[1:] for line in cmake.splitlines() if line.startswith("+") and not line.startswith("+++")]
    # the five ExternalProjects and upstream's install of their output are gone; nothing installs a skel
    assert removed[:5] == [f"build_htp_skel({a})" for a in ARCHS]
    assert "install(FILES ${HTP_SKELS} TYPE LIB)" in removed
    assert not any("build_htp_skel(" in line or "HTP_SKELS" in line or ".so.dsp" in line or "llama_cpp/htp" in line
                   for line in added if not line.lstrip().startswith("#"))
    assert "install(FILES ${CMAKE_CURRENT_BINARY_DIR}/npu-backend.txt DESTINATION llama_cpp)" in added
    assert wh.NPU_MARKER == "llama_cpp/npu-backend.txt"
    cpp = hunks["vendor/llama.cpp/ggml/src/ggml-hexagon/ggml-hexagon.cpp"]
    assert "+GGML_BACKEND_API int glossarion_hexagon_session_check(size_t index, char * err, size_t err_len);" in cpp
    assert "+        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);" in cpp
    assert "+    } catch (const std::exception & exc) {" in cpp
    loader = hunks["llama_cpp/_ctypes_extensions.py"]
    assert '"glossarion_hexagon_session_check"' in loader
    # the packaged-skeleton code is gone
    for gone in ("_htp_source", "_copy_skeletons", "_packaged_skeletons", ".so.dsp", "_HTP_SOURCE"):
        assert gone not in loader, gone
    # never through ggml_backend_load_all (only comments name it)
    calls = [line for line in loader.splitlines() if line.startswith("+")
             and re.search(r'^\+[^#"]*ggml_backend_load_all\(', line)]
    assert calls == []


def test_loader_steps_match_the_app(monkeypatch):
    ns = _patched_cpu_loader(_real_ctypes_ns())
    assert ns["NPU_STEPS"] == ("shipped", "libcdsprpc", "arch", "skeletons", "registry", "session")
    assert ns["NPU_BACKEND"] == "HTP" and ns["NPU_LIBRARY"] == "libggml-hexagon.so"
    monkeypatch.syspath_prepend(str(MOBILE / "app"))
    try:
        from glossarion_mobile.services.local_llm import engine
    except Exception:  # pragma: no cover - the app package is not importable here
        pytest.skip("glossarion_mobile not importable")
    if hasattr(engine, "NPU_STEPS"):
        assert tuple(engine.NPU_STEPS) == ns["NPU_STEPS"]
        assert engine.NPU_BACKEND == ns["NPU_BACKEND"]


# ---- the APK ------------------------------------------------------------------------------------------------------

MANIFEST_XML = """N: android=http://schemas.android.com/apk/res/android
  E: manifest (line=2)
    E: application (line=10)
      E: uses-native-library (line=12)
        A: http://schemas.android.com/apk/res/android:name(0x01010003)="libOpenCL.so" (Raw: "libOpenCL.so")
        A: http://schemas.android.com/apk/res/android:required(0x0101028e)=false
      E: uses-native-library (line=15)
        A: http://schemas.android.com/apk/res/android:name(0x01010003)="libcdsprpc.so" (Raw: "libcdsprpc.so")
        A: http://schemas.android.com/apk/res/android:required(0x0101028e)=false
"""


def test_verify_apk_checks_the_npu_backend():
    va = _verify_apk()
    tree = va.parse_xmltree(MANIFEST_XML)
    names = ["lib/arm64-v8a/libggml-hexagon.so", "lib/arm64-v8a/libllama.so"]
    assert va.npu_problems(tree, names, [], True) == []
    assert va.npu_problems(tree, ["lib/x86_64/libllama.so"], [], True) == []    # the x86_64 split: no NPU backend
    # [hexagon] enabled: an arm64 APK without the backend fails
    assert va.npu_problems(tree, ["lib/arm64-v8a/libllama.so"], [], True) == [
        "lib/arm64-v8a/libggml-hexagon.so is missing although wheelhouse.toml [hexagon] is enabled"]
    no_entry = va.parse_xmltree(MANIFEST_XML.split("      E: uses-native-library (line=15)")[0])
    assert "no <uses-native-library android:name=\"libcdsprpc.so\">" in va.npu_problems(no_entry, names, [], True)[0]
    required = va.parse_xmltree(MANIFEST_XML.replace("required(0x0101028e)=false\n", "required(0x0101028e)=true\n"))
    assert any('libcdsprpc.so> is required="true"' in p for p in va.npu_problems(required, names, [], True))
    bundled = va.npu_problems(tree, names + ["lib/arm64-v8a/libcdsprpc.so"], [], True)
    assert any("bundles ['lib/arm64-v8a/libcdsprpc.so']" in p for p in bundled)
    # a skeleton anywhere fails, on or off
    skeleton = ["assets/app/sitepackages.zip!llama_cpp/htp/libggml-htp-v79.so.dsp"]
    for enabled in (True, False):
        problems = va.npu_problems(tree, names if enabled else ["lib/arm64-v8a/libllama.so"], skeleton, enabled)
        assert problems == [f"the APK carries HTP skeleton files {skeleton}: they may not ship (Hexagon SDK licence); "
                            f"the app downloads them from llama.cpp's release"]
    # [hexagon] switched off: no NPU pieces at all
    assert va.npu_problems(no_entry, ["lib/arm64-v8a/libllama.so"], [], False) == []
    assert va.npu_problems(tree, names, [], False) == [
        "lib/arm64-v8a/libggml-hexagon.so ships but wheelhouse.toml [hexagon] is switched off"]


def test_verify_apk_finds_skeletons_in_jnilibs_and_the_packaged_site_packages(tmp_path):
    va = _verify_apk()
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_STORED) as z:
        z.writestr("llama_cpp/htp/libggml-htp-v79.so.dsp", make_elf32())
        z.writestr("llama_cpp/data.bin", make_elf32())                 # a Hexagon ELF under another name
        z.writestr("llama_cpp/npu-backend.txt", "libggml-hexagon.so\n")
        z.writestr("llama_cpp/__init__.pyc", b"x")
    apk = tmp_path / "app.apk"
    with zipfile.ZipFile(apk, "w") as z:
        z.writestr("lib/arm64-v8a/libggml-hexagon.so", make_elf())
        z.writestr("lib/arm64-v8a/libggml-htp-v81.so", make_elf32())
        z.writestr("assets/flutter_assets/app/sitepackages.zip", inner.getvalue())
        z.writestr("assets/flutter_assets/app/other.zip", b"not a zip")
        z.writestr("assets/flutter_assets/app/sitepackages-x.zip", b"not a zip either")
    with zipfile.ZipFile(apk) as z:
        assert va.skeleton_files(z) == [
            "assets/flutter_assets/app/sitepackages.zip!llama_cpp/data.bin",
            "assets/flutter_assets/app/sitepackages.zip!llama_cpp/htp/libggml-htp-v79.so.dsp",
            "lib/arm64-v8a/libggml-htp-v81.so"]
    clean = tmp_path / "clean.apk"
    with zipfile.ZipFile(clean, "w") as z:
        z.writestr("lib/arm64-v8a/libggml-hexagon.so", make_elf())
        z.writestr("assets/flutter_assets/app/sitepackages.zip", b"")
    with zipfile.ZipFile(clean) as z:
        assert va.skeleton_files(z) == []


def test_verify_apk_fails_a_bundle_that_carries_skeletons(tmp_path, monkeypatch):
    va = _verify_apk()
    monkeypatch.setattr(va.shutil, "which", lambda name: None)          # no jarsigner: the signature is skipped
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("llama_cpp/htp/libggml-htp-v79.so.dsp", make_elf32())
    args = argparse.Namespace(allowed_abis=["arm64-v8a"], expect_release=False)

    def bundle(name, members):
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("base/manifest/AndroidManifest.xml", b"")
            z.writestr("base/lib/arm64-v8a/libllama.so", make_elf())
            for member, data in members.items():
                z.writestr(member, data)
        report = va.Report(path)
        va.check_aab(report, args)
        return report.errors

    assert bundle("clean.aab", {}) == []
    errors = bundle("bad.aab", {"base/assets/flutter_assets/app/sitepackages.zip": inner.getvalue(),
                                "base/lib/arm64-v8a/libggml-htp-v81.so": make_elf32()})
    found = ["base/assets/flutter_assets/app/sitepackages.zip!llama_cpp/htp/libggml-htp-v79.so.dsp",
             "base/lib/arm64-v8a/libggml-htp-v81.so"]
    assert errors == va.skeleton_problems(found, "bundle")
    assert errors[0].startswith("the bundle carries HTP skeleton files")


def test_verify_apk_reads_the_hexagon_switch(tmp_path):
    va = _verify_apk()
    assert va.DEFAULT_WHEELHOUSE == wh.DEFAULT_MANIFEST
    assert va.hexagon_enabled(va.DEFAULT_WHEELHOUSE) is (wh.hexagon_spec(wh.Ctx(wh.DEFAULT_MANIFEST, "android").data)
                                                         is not None)
    assert va.hexagon_enabled(_manifest(tmp_path, enabled=False)) is False
    assert va.hexagon_enabled(_manifest(tmp_path / "on", enabled=True)) is True
    assert va.hexagon_enabled(tmp_path / "missing.toml") is False and va.hexagon_enabled(None) is False


def test_native_extension_manifest_declares_fastrpc_optional():
    manifest = (MOBILE / "extensions" / "flet_glossarion_native" / "src" / "flutter" / "flet_glossarion_native" /
                "android" / "src" / "main" / "AndroidManifest.xml").read_text(encoding="utf-8")
    assert re.search(r'<uses-native-library\s+android:name="libcdsprpc.so"\s+android:required="false"\s*/>', manifest)


def test_workflow_caches_the_sdk_under_its_sha256_and_rebuilds_the_engine_on_a_recipe_change():
    workflow = MOBILE.parents[1] / ".github" / "workflows" / "mobile-wheels.yml"
    if not workflow.is_file():
        pytest.skip("not a full repository checkout")
    text = workflow.read_text(encoding="utf-8")
    assert text.count("key: mobile-hexagon-sdk-${{ steps.pins.outputs.hexagon_sdk_sha256 }}") == 2
    assert "HEXAGON_SDK_DOWNLOADS: ${{ runner.temp }}/hexagon-sdk" in text
    assert "path: ${{ runner.temp }}/hexagon-sdk" in text
    # the engine wheel's cache key hashes the manifest, the build scripts and the whole recipe (mobile.patch
    # included), so a patch or pin change rebuilds it
    keys = re.findall(r"key: (mobile-wheels-v\d+-\$\{\{ inputs\.platform \}\}-llama-cpp-python-\$\{\{ hashFiles\((.*?)\) \}\})",
                      text)
    assert len(keys) == 2 and len({k for k, _ in keys}) == 1, keys
    hashed = {p.strip().strip("'") for p in keys[0][1].split(",")}
    assert {"src/mobile/ci/wheels/wheelhouse.toml", "src/mobile/ci/wheels/build_wheels.sh",
            "src/mobile/ci/wheels/wheels.py", "src/mobile/ci/wheels/recipes/llama-cpp-python/**"} <= hashed
    build = (MOBILE / "ci" / "wheels" / "build_wheels.sh").read_text(encoding="utf-8")
    assert 'HEXAGON_SDK_DOWNLOADS="${HEXAGON_SDK_DOWNLOADS:-$FORGE_DIR/downloads}"' in build
    assert "stage-hexagon --platform android" in build
