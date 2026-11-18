"""The on-device engine's optional NPU backend in the build (wheelhouse.toml [hexagon], recipes/llama-cpp-python,
mobile.patch 10., ci/wheels/wheels.py, ci/verify_apk.py): the manifest switch, the SDK staging with a tiny fake
SDK, the CMake initial cache the arm64 recipe loads, the verify rules for libggml-hexagon and the HTP skeletons
(fake arm64 / QDSP6 ELF files), the patched loader's register_npu_backend / device_index over a fake ctypes, and
the APK checks. No network, no SDK, no native NPU.
"""

from __future__ import annotations

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


# ---- the manifest switch ---------------------------------------------------------------------------------------

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
    assert wh.hexagon_targets(ctx) == {"arm64-v8a": ARCHS}
    assert wh.hexagon_targets(wh.Ctx(wh.DEFAULT_MANIFEST, "ios")) == {"arm64-v8a": ARCHS}   # android targets only
    assert wh.needs_hexagon_sdk(ctx, ["llama-cpp-python"]) and not wh.needs_hexagon_sdk(ctx, ["pillow"])
    assert not wh.needs_hexagon_sdk(wh.Ctx(wh.DEFAULT_MANIFEST, "ios"), ["llama-cpp-python"])


def test_switched_off_manifest_expects_and_fetches_nothing(tmp_path):
    ctx = wh.Ctx(_manifest(tmp_path, enabled=False), "android", wh.DEFAULT_PYPROJECT)
    assert wh.hexagon_spec(ctx.data) is None and wh.hexagon_targets(ctx) == {}
    assert wh.builds_hexagon_recipe(ctx, ["llama-cpp-python"])          # the recipe still loads the cache (OFF)
    assert not wh.needs_hexagon_sdk(ctx, ["llama-cpp-python"])
    plan = {Path(p).name for _u, p, _s in wh.fetch_plan(ctx, ["llama-cpp-python"], tmp_path / "dl")}
    assert not any(n.startswith("hexagon-sdk") for n in plan)
    assert wh.profile_libs(wh.SITE_PROFILES["llama-cpp-python"], "arm64-v8a", wh.hexagon_targets(ctx)) == \
        wh.profile_libs(wh.SITE_PROFILES["llama-cpp-python"], "arm64-v8a")


def test_fetch_plan_puts_the_sdk_where_ci_caches_it(tmp_path):
    ctx = wh.Ctx(wh.DEFAULT_MANIFEST, "android")
    sdk = wh.hexagon_spec(ctx.data)["sdk"]
    plan = wh.fetch_plan(ctx, ["llama-cpp-python"], tmp_path / "dl", tmp_path / "cache")
    assert (sdk["url"], tmp_path / "cache" / sdk["file"], sdk["sha256"]) in plan
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


@pytest.mark.parametrize("change, message", [
    ({"enabled": '"yes"'}, "needs enabled = true or false"),
    ({"abis": '["arm64-v8a", "x86_64"]'}, "abis must be ['arm64-v8a']"),
    ({"htp_archs": '["73"]'}, "htp_archs must be distinct HTP arch names"),
    ({"htp_archs": "[]"}, "htp_archs must be distinct HTP arch names"),
    ({"tools": '"latest"'}, "tools must be a version string"),
    ({"sdk": '{ url = "http://x/y.tar.xz", sha256 = "' + "0" * 64 + '", file = "y.tar.xz" }'}, "an https url"),
    ({"sdk": '{ url = "https://x/y.tar.xz", file = "y.tar.xz" }'}, "needs sdk = {url, sha256, file}"),
])
def test_bad_hexagon_tables_are_refused(change, message):
    table = {"enabled": "true", "abis": '["arm64-v8a"]', "version": '"6.7.0.0"', "tools": '"21.0.04"',
             "htp_archs": '["v79"]', "sdk": '{ url = "https://x/y.tar.xz", sha256 = "' + "a" * 64 + '", file = "y.tar.xz" }'}
    table.update(change)
    data = wh.cmw.tomllib.loads("[hexagon]\n" + "".join(f"{k} = {v}\n" for k, v in table.items()))
    with pytest.raises(wh.WheelsError, match=re.escape(message)):
        wh.hexagon_spec(data)
    assert wh.hexagon_spec({}) is None and wh.hexagon_spec({"hexagon": {"enabled": False}}) is None


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


def make_npu_android(sp: Path, archs=ARCHS, skeletons=None, libs=HEX_LIBS_ARM64, **kw) -> Path:
    """build/site-packages/arm64-v8a with the engine, libggml-hexagon and the HTP skeletons
    (``skeletons``: {file name: bytes}, default one QDSP6 file per arch)."""
    arch = make_llama_android(sp, "arm64-v8a", libs=libs, **kw)
    htp = arch / "llama_cpp" / "htp"
    htp.mkdir()
    for name, data in (skeletons if skeletons is not None else
                       {f"libggml-htp-{a}.so.dsp": make_elf32(extra=a.encode()) for a in archs}).items():
        (htp / name).write_bytes(data)
    return arch


HEXAGON = {"arm64-v8a": ARCHS}


def test_verify_site_packages_accepts_the_npu_backend(tmp_path):
    sp = tmp_path / "sp"
    make_npu_android(sp)
    errors, notes = _verify_llama(sp, "android", make_ndk(tmp_path / "ndk", LLAMA_NDK_EXPORTS), hexagon=HEXAGON)
    assert errors == []
    assert any(re.search(r"arm64-v8a: llama_cpp/htp/libggml-htp-v85\.so\.dsp Hexagon DSP ELF; [\d.]+ MB", n)
               for n in notes), notes
    assert any("lib/libggml-hexagon.so NEEDED" in n for n in notes)
    assert "arm64-v8a: lib/libggml-hexagon.so armv8.0 only: optional instructions none" in notes


@pytest.mark.parametrize("kwargs, hexagon, message", [
    # a skeleton missing / one too many
    ({"archs": ARCHS[:-1]}, HEXAGON, "llama_cpp/htp holds ['libggml-htp-v73.so.dsp', 'libggml-htp-v75.so.dsp', "
                                     "'libggml-htp-v79.so.dsp', 'libggml-htp-v81.so.dsp'], expected exactly"),
    ({"archs": ARCHS + ("v89",)}, HEXAGON, "'libggml-htp-v89.so.dsp'], expected exactly"),
    # an ARM library where a DSP skeleton belongs
    ({"skeletons": {**{f"libggml-htp-{a}.so.dsp": make_elf32() for a in ARCHS[1:]},
                    "libggml-htp-v73.so.dsp": make_elf()}}, HEXAGON,
     "llama_cpp/htp/libggml-htp-v73.so.dsp: ELF machine 183, not Hexagon (EM_QDSP6 164)"),
    ({"skeletons": {**{f"libggml-htp-{a}.so.dsp": make_elf32() for a in ARCHS[1:]},
                    "libggml-htp-v73.so.dsp": b"text"}}, HEXAGON, "libggml-htp-v73.so.dsp: not an ELF file"),
    # the backend library missing
    ({"libs": LLAMA_LIBS_ARM64}, HEXAGON, "expected exactly ['libggml', 'libggml-base', 'libggml-cpu-android_armv8.0_1'"),
    # [hexagon] switched off, but the build still carries it
    ({}, None, "llama_cpp/htp holds ['libggml-htp-v73.so.dsp'"),
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


def test_a_skeleton_shipped_as_a_library_is_refused(tmp_path):
    sp = tmp_path / "sp"
    arch = make_npu_android(sp)
    (arch / "llama_cpp" / "lib" / "libggml-htp-v79.so").write_bytes(make_elf32())
    errors, _ = _verify_llama(sp, "android", make_ndk(tmp_path / "ndk", LLAMA_NDK_EXPORTS), hexagon=HEXAGON)
    text = "\n".join(errors)
    assert "HTP skeleton files outside llama_cpp/htp/libggml-htp-<arch>.so.dsp: ['llama_cpp/lib/libggml-htp-v79.so']" \
        in text
    assert wh.stray_htp_files(["llama_cpp/htp/libggml-htp-v79.so.dsp", "llama_cpp/htp/libggml-htp-v79.so",
                               "x/libggml-htp-v79.so.dsp", "llama_cpp/lib/libggml.so"]) == \
        ["llama_cpp/htp/libggml-htp-v79.so", "x/libggml-htp-v79.so.dsp"]


def test_switched_off_build_without_npu_files_passes(tmp_path):
    sp = tmp_path / "sp"
    make_llama_android(sp, "arm64-v8a")
    errors, _ = _verify_llama(sp, "android", make_ndk(tmp_path / "ndk", LLAMA_NDK_EXPORTS), hexagon={})
    assert errors == []
    # ... but with [hexagon] on, the same build is missing the backend and its skeletons
    errors, _ = _verify_llama(sp, "android", make_ndk(tmp_path / "ndk2", LLAMA_NDK_EXPORTS), hexagon=HEXAGON)
    text = "\n".join(errors)
    assert "llama_cpp/htp holds [], expected exactly ['libggml-htp-v73.so.dsp'" in text and "libggml-hexagon" in text


def test_verify_native_checks_the_npu_files_of_the_wheel(tmp_path):
    ndk = make_ndk(tmp_path / "ndk", LLAMA_NDK_EXPORTS)
    d = ndk / "toolchains" / "llvm" / "prebuilt" / "linux-x86_64" / "sysroot" / "usr" / "lib" / "aarch64-linux-android"
    (d / "libc++_shared.so").write_bytes(make_elf(needed=("libc.so",), symbols=[("_ZNSt6__ndk14cout", base.GLOBAL,
                                                                                   base.DEF)]))
    stubs = wh.NdkStubs(ndk)
    wheel = _zip_tree(make_npu_android(tmp_path / "sp"), tmp_path / "llama_cpp_python-0.3.37-1000-cp313-cp313-a.whl")
    errors, notes = wh.verify_native_wheel(wheel, "llama-cpp-python", "arm64-v8a", "3.5.9", (13, 0), stubs, HEXAGON)
    assert errors == [], errors
    assert any("llama_cpp/htp/libggml-htp-v79.so.dsp Hexagon DSP ELF" in n for n in notes)
    # the switched-off expectation refuses the same wheel; a wheel without skeletons fails the enabled one
    errors, _ = wh.verify_native_wheel(wheel, "llama-cpp-python", "arm64-v8a", "3.5.9", (13, 0), stubs, {})
    assert any("([hexagon] is switched off)" in e for e in errors), errors
    bare = _zip_tree(make_llama_android(tmp_path / "sp2", "arm64-v8a"), tmp_path / "bare.whl")
    errors, _ = wh.verify_native_wheel(bare, "llama-cpp-python", "arm64-v8a", "3.5.9", (13, 0), stubs, HEXAGON)
    assert any("llama_cpp/htp holds [], expected exactly" in e for e in errors), errors
    # x86_64 never carries it, whatever the manifest says about arm64
    x86 = _zip_tree(make_llama_android(tmp_path / "sp3", "x86_64"), tmp_path / "x86.whl")
    (d.parent / "x86_64-linux-android").mkdir(exist_ok=True)
    (d.parent / "x86_64-linux-android" / "libc++_shared.so").write_bytes(make_elf(
        machine=62, needed=("libc.so",), symbols=[("_ZNSt6__ndk14cout", base.GLOBAL, base.DEF)]))
    errors, _ = wh.verify_native_wheel(x86, "llama-cpp-python", "x86_64", "3.5.9", (13, 0), stubs, HEXAGON)
    assert errors == [], errors


# ---- the patched loader: register_npu_backend / device_index over a fake ctypes -----------------------------------

ARCH_V79 = 0x79  # DSPRPC_GET_DSP_INFO ARCH_VER reports the arch as BCD-like nibbles (htp-drv.cpp)


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


def _loader(tmp_path, monkeypatch, skeleton_archs=(73, 75, 79, 81, 85), **kw):
    for key in ("GGML_HEXAGON_LIBHTP_DIR", "GGML_HEXAGON_ARCH", "GGML_HEXAGON_POWER_MODE"):
        monkeypatch.setenv(key, "")          # restored after the test; the loader sets them
    ns_ctypes, libggml, calls = _npu_ctypes(tmp_path, **kw)
    ns = _patched_cpu_loader(ns_ctypes)
    htp = tmp_path / "pkg" / "llama_cpp" / "htp"
    htp.mkdir(parents=True, exist_ok=True)
    for a in skeleton_archs:
        (htp / f"libggml-htp-v{a}.so.dsp").write_bytes(b"\x7fELF-v%d" % a)
    lib = tmp_path / "pkg" / "llama_cpp" / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    (lib / "libggml-hexagon.so").write_bytes(b"")
    ns["_HTP_SOURCE"], ns["_LIBGGML"], ns["_BASE_PATH"] = htp, libggml, lib
    return ns, calls


def _steps(state):
    return [(s["name"], s["ok"]) for s in state["steps"]]


def test_register_npu_backend_sets_the_environment_before_the_only_load(tmp_path, monkeypatch):
    ns, calls = _loader(tmp_path, monkeypatch)
    assert ns["NPU_STATE"] == {} and calls["load"] == []                        # nothing loads on import
    assert ns["npu_backend_shipped"]() is True and calls["cdll"] == []           # naming it loads nothing
    skel = tmp_path / "files" / "htp"
    state = ns["register_npu_backend"](str(skel), {"GGML_HEXAGON_POWER_MODE": "balanced", "PATH": "/x",
                                                   "GGML_HEXAGON_LIBHTP_DIR": "/elsewhere"})
    assert state["backend"] == "HTP" and state["device"] == "HTP0" and state["arch"] == 79 and state["error"] == ""
    assert _steps(state) == [(n, True) for n in ("shipped", "libcdsprpc", "arch", "skeletons", "registry", "session")]
    assert "Hexagon v79; unsigned PD supported" in state["steps"][2]["detail"]
    assert "ignored options ['PATH', 'GGML_HEXAGON_LIBHTP_DIR']" in state["steps"][4]["detail"]
    # our own DSPRPC_GET_DSP_INFO queries (cDSP domain 3): ARCH_VER (6), then UNSIGNED_PD_SUPPORT (1)
    assert calls["control"] == [(2, 3, 6, 12), (2, 3, 1, 12)]
    # the skeletons, copied out of the package as libggml-htp-vNN.so
    assert sorted(p.name for p in skel.iterdir()) == [f"libggml-htp-v{a}.so" for a in (73, 75, 79, 81, 85)]
    assert (skel / "libggml-htp-v79.so").read_bytes() == b"\x7fELF-v79"
    # loaded once, by the file next to libggml, with the environment already set
    assert calls["load"] == [str(tmp_path / "pkg" / "llama_cpp" / "lib" / "libggml-hexagon.so")]
    assert calls["env_at_load"] == [{"GGML_HEXAGON_LIBHTP_DIR": str(skel), "GGML_HEXAGON_ARCH": "79",
                                     "GGML_HEXAGON_POWER_MODE": "balanced"}]
    assert calls["session"] == [0]
    # idempotent: the first answer, no second load
    again = ns["register_npu_backend"](str(tmp_path / "other"), {})
    assert again == state and len(calls["load"]) == 1 and not (tmp_path / "other").exists()
    again["steps"][0]["ok"] = False
    assert ns["register_npu_backend"](str(skel))["steps"][0]["ok"] is True      # callers get copies


def test_skeleton_copies_are_refreshed_only_when_different(tmp_path, monkeypatch):
    skel = tmp_path / "files" / "htp"
    skel.mkdir(parents=True)
    (skel / "libggml-htp-v79.so").write_bytes(b"\x7fELF-v79")      # up to date
    (skel / "libggml-htp-v81.so").write_bytes(b"old build")         # stale
    ns, _calls = _loader(tmp_path, monkeypatch)
    state = ns["register_npu_backend"](str(skel))
    assert "(3 copied, 2 up to date)" not in state["steps"][3]["detail"]
    assert "libggml-htp-v79.so in" in state["steps"][3]["detail"] and "(4 copied, 1 up to date)" in \
        state["steps"][3]["detail"]
    assert (skel / "libggml-htp-v81.so").read_bytes() == b"\x7fELF-v81"


@pytest.mark.parametrize("kw, failed, detail", [
    ({"rpc": False}, "libcdsprpc", 'no FastRPC driver (dlopen failed: library "libcdsprpc.so" not found)'),
    ({"rpc_control": False}, "arch", "libcdsprpc.so has no remote_handle_control"),
    ({"control_err": -2147482624}, "arch", "remote_handle_control(DSPRPC_GET_DSP_INFO) failed (error 0x80000400)"),
    ({"arch_value": 0}, "arch", "the cDSP reported arch value 0x0"),
    ({"arch_value": 0x68}, "skeletons", "no HTP skeleton for v68 (this build has v73, v75, v79, v81, v85)"),
    ({"load_ok": False}, "registry", "the Hexagon backend did not start (ggml_backend_init returned NULL"),
    ({"devices": ()}, "registry", "HTP registered without a device"),
    ({"session": -1}, "session", "ggml-hex: remote_session_control(unsign) failed (see log for details)"),
    ({"session_symbol": False}, "session", "not checked: this build has no session check"),
])
def test_register_npu_backend_reports_the_first_failure_and_never_raises(tmp_path, monkeypatch, kw, failed, detail):
    ns, calls = _loader(tmp_path, monkeypatch, **kw)
    state = ns["register_npu_backend"](str(tmp_path / "files" / "htp"))
    names = [s["name"] for s in state["steps"]]
    assert names == ["shipped", "libcdsprpc", "arch", "skeletons", "registry", "session"]
    first = next(s for s in state["steps"] if not s["ok"])
    assert first["name"] == failed and detail in first["detail"], state
    assert state["error"] == first["detail"]
    after = state["steps"][names.index(failed) + 1:]
    assert all(not s["ok"] and s["detail"] == "not checked (an earlier step failed)" for s in after)
    if names.index(failed) < names.index("registry"):
        assert calls["load"] == []                                  # nothing loads after an early failure


def test_register_npu_backend_without_skeletons_or_libraries(tmp_path, monkeypatch):
    ns, calls = _loader(tmp_path, monkeypatch, skeleton_archs=())
    assert ns["npu_backend_shipped"]() is False
    state = ns["register_npu_backend"](str(tmp_path / "s"))
    assert _steps(state)[0] == ("shipped", False) and "carries no HTP skeletons" in state["error"]
    assert calls["cdll"] == []
    # the engine's libraries not loaded yet: the registry step says so, nothing raises
    ns, calls = _loader(tmp_path / "b", monkeypatch)
    ns["_LIBGGML"] = None
    state = ns["register_npu_backend"](str(tmp_path / "b" / "s"))
    assert ("registry", False) in _steps(state) and state["error"] == "the engine's libraries are not loaded"
    # an unwritable skeleton directory (a file there): reported on the skeletons step
    ns, _calls = _loader(tmp_path / "c", monkeypatch)
    (tmp_path / "c" / "blocked").write_text("x", encoding="utf-8")
    state = ns["register_npu_backend"](str(tmp_path / "c" / "blocked"))
    bad = next(s for s in state["steps"] if not s["ok"])
    assert bad["name"] == "skeletons" and bad["detail"].startswith(("FileExistsError", "NotADirectoryError", "OSError"))


def test_npu_backend_shipped_on_android_arm64_without_a_file(tmp_path, monkeypatch):
    ns, _calls = _loader(tmp_path, monkeypatch)
    ns["_BASE_PATH"] = tmp_path / "sitepackages.zip" / "llama_cpp" / "lib"      # jniLibs: no file on disk
    monkeypatch.setattr(ns["sys"], "platform", "android")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    assert ns["npu_backend_shipped"]() is True
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    assert ns["npu_backend_shipped"]() is False
    state = ns["register_npu_backend"](str(tmp_path / "s"))
    assert _steps(state)[0] == ("shipped", False)


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
    ns = _patched_cpu_loader(types.SimpleNamespace(**{k: getattr(real_ctypes, k) for k in dir(real_ctypes)
                                                      if not k.startswith("__")}))
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

def test_patch_installs_the_skeletons_as_package_data_and_adds_the_session_check():
    text = PATCH.read_text(encoding="utf-8")
    assert "\r" not in text
    assert "10. Optional NPU backend (Qualcomm Hexagon HTP, ggml-hexagon)" in text.split("diff --git", 1)[0]
    hunks = {m.group(1): m.group(2) for m in re.finditer(r"^diff --git a/(\S+) b/\S+\n(.*?)(?=^diff --git |\Z)", text,
                                                         re.M | re.S)}
    cmake = hunks["vendor/llama.cpp/ggml/src/ggml-hexagon/CMakeLists.txt"]
    assert "-install(FILES ${HTP_SKELS} TYPE LIB)" in cmake
    assert "+    install(FILES ${skel} DESTINATION llama_cpp/htp RENAME ${skel_name}.dsp)" in cmake
    assert wh.HTP_DIR == "llama_cpp/htp" and wh.HTP_FILE.match("libggml-htp-v79.so.dsp")
    cpp = hunks["vendor/llama.cpp/ggml/src/ggml-hexagon/ggml-hexagon.cpp"]
    assert "+GGML_BACKEND_API int glossarion_hexagon_session_check(size_t index, char * err, size_t err_len);" in cpp
    assert "+        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);" in cpp
    assert "+    } catch (const std::exception & exc) {" in cpp
    loader = hunks["llama_cpp/_ctypes_extensions.py"]
    assert '"glossarion_hexagon_session_check"' in loader
    # never through ggml_backend_load_all (only comments name it)
    calls = [line for line in loader.splitlines() if line.startswith("+")
             and re.search(r'^\+[^#"]*ggml_backend_load_all\(', line)]
    assert calls == []


def test_patch_keeps_the_android_flags_out_of_the_skeleton_sub_build():
    # forge exports the Android CFLAGS/LDFLAGS for the whole wheel build; CMake would seed each htp-vNN
    # ExternalProject's base flags from them (the toolchain file sets only the per-config flags, with -Werror).
    text = PATCH.read_text(encoding="utf-8")
    cmake = re.search(r"^diff --git a/vendor/llama.cpp/ggml/src/ggml-hexagon/CMakeLists.txt .*?(?=^diff --git |\Z)",
                      text, re.M | re.S).group(0)
    lines = cmake.splitlines()
    assert '-            -DPREBUILT_LIB_DIR="toolv21_${V}")' in lines
    start = lines.index('+            -DPREBUILT_LIB_DIR="toolv21_${V}"')
    end = next(i for i in range(start, len(lines)) if lines[i].endswith(")"))
    args = [line[1:].strip() for line in lines[start + 1:end + 1] if not line[1:].strip().startswith("#")]
    assert all(line.startswith("+") for line in lines[start:end + 1])
    assert args[-1].endswith(")")
    blanked = {a.rstrip(")") for a in args}
    assert blanked == {f"-DCMAKE_{v}_FLAGS=" for v in ("C", "CXX", "ASM", "SHARED_LINKER", "MODULE_LINKER",
                                                       "EXE_LINKER")}


def test_loader_steps_match_the_app(monkeypatch):
    ns = _patched_cpu_loader(types.SimpleNamespace(**{k: getattr(real_ctypes, k) for k in dir(real_ctypes)
                                                      if not k.startswith("__")}))
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
    packaged = [f"llama_cpp/htp/libggml-htp-{a}.so.dsp" for a in ARCHS] + ["llama_cpp/__init__.pyc"]
    assert va.npu_problems(tree, names, packaged) == []
    assert va.npu_problems(tree, ["lib/x86_64/libllama.so"], []) == []      # no NPU backend: nothing to check
    no_entry = va.parse_xmltree(MANIFEST_XML.split("      E: uses-native-library (line=15)")[0])
    assert "no <uses-native-library android:name=\"libcdsprpc.so\">" in va.npu_problems(no_entry, names, packaged)[0]
    required = va.parse_xmltree(MANIFEST_XML.replace("required(0x0101028e)=false\n", "required(0x0101028e)=true\n"))
    assert any('libcdsprpc.so> is required="true"' in p for p in va.npu_problems(required, names, packaged))
    bundled = va.npu_problems(tree, names + ["lib/arm64-v8a/libcdsprpc.so"], packaged)
    assert any("bundles ['lib/arm64-v8a/libcdsprpc.so']" in p for p in bundled)
    diverted = va.npu_problems(tree, names + ["lib/arm64-v8a/libggml-htp-v79.so"], packaged)
    assert any("HTP skeletons in jniLibs ['lib/arm64-v8a/libggml-htp-v79.so']" in p for p in diverted)
    assert any("has no llama_cpp/htp/libggml-htp-vNN.so.dsp" in p for p in va.npu_problems(tree, names, []))


def test_verify_apk_reads_the_packaged_site_packages(tmp_path):
    va = _verify_apk()
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_STORED) as z:
        z.writestr("llama_cpp/htp/libggml-htp-v79.so.dsp", make_elf32())
        z.writestr("llama_cpp/__init__.pyc", b"x")
    apk = tmp_path / "app.apk"
    with zipfile.ZipFile(apk, "w") as z:
        z.writestr("lib/arm64-v8a/libggml-hexagon.so", b"\x7fELF")
        z.writestr("assets/flutter_assets/app/sitepackages.zip", inner.getvalue())
        z.writestr("assets/flutter_assets/app/other.zip", b"not a zip")
    with zipfile.ZipFile(apk) as z:
        assert va.packaged_python_names(z) == ["llama_cpp/htp/libggml-htp-v79.so.dsp", "llama_cpp/__init__.pyc"]


def test_native_extension_manifest_declares_fastrpc_optional():
    manifest = (MOBILE / "extensions" / "flet_glossarion_native" / "src" / "flutter" / "flet_glossarion_native" /
                "android" / "src" / "main" / "AndroidManifest.xml").read_text(encoding="utf-8")
    assert re.search(r'<uses-native-library\s+android:name="libcdsprpc.so"\s+android:required="false"\s*/>', manifest)


def test_workflow_caches_the_sdk_under_its_sha256():
    workflow = MOBILE.parents[1] / ".github" / "workflows" / "mobile-wheels.yml"
    if not workflow.is_file():
        pytest.skip("not a full repository checkout")
    text = workflow.read_text(encoding="utf-8")
    assert text.count("key: mobile-hexagon-sdk-${{ steps.pins.outputs.hexagon_sdk_sha256 }}") == 2
    assert "HEXAGON_SDK_DOWNLOADS: ${{ runner.temp }}/hexagon-sdk" in text
    assert "path: ${{ runner.temp }}/hexagon-sdk" in text
    build = (MOBILE / "ci" / "wheels" / "build_wheels.sh").read_text(encoding="utf-8")
    assert 'HEXAGON_SDK_DOWNLOADS="${HEXAGON_SDK_DOWNLOADS:-$FORGE_DIR/downloads}"' in build
    assert "stage-hexagon --platform android" in build
