"""The on-device engine's optional GPU backends in the build (recipes/llama-cpp-python, mobile.patch 8.,
ci/wheels/wheels.py, ci/verify_apk.py): the CMake flags per target, the patched loader's on-demand
registration (fake ctypes), the verify profile's optional-library rules, the link-time OpenCL staging
helpers and the APK manifest check.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tarfile
import types
from pathlib import Path

import pytest

import test_check_mobile_wheels_wheelhouse as base
from test_check_mobile_wheels_wheelhouse import (GLOBAL, LLAMA_CODE_ARM64, LLAMA_NDK_EXPORTS, MOBILE, UND, _FakeFn,
                                                 _patched_cpu_loader, _render_recipe, _verify_llama, cmw,
                                                 make_elf, make_llama_android, make_ndk, wh)

RECIPE = MOBILE / "ci" / "wheels" / "recipes" / "llama-cpp-python" / "meta.yaml"
PATCH = RECIPE.parent / "patches" / "mobile.patch"


def test_recipe_builds_opencl_on_android_arm64_and_metal_on_ios():
    gpu = ("-DGGML_OPENCL=ON", "-DGGML_OPENCL_EMBED_KERNELS=ON", "-DGGML_OPENCL_USE_ADRENO_KERNELS=ON")
    for sdk, arch in (("android", "arm64-v8a"), ("android", "x86_64"), ("iphoneos", "arm64"),
                      ("iphonesimulator", "arm64"), ("iphonesimulator", "x86_64")):
        args = _render_recipe(RECIPE, sdk, arch)["build"]["script_env"]["CMAKE_ARGS"].split()
        opencl = [a for a in args if a.startswith("-DGGML_OPENCL=")]
        assert len(opencl) == 1, (sdk, arch, args)                    # one value, no override games
        if (sdk, arch) == ("android", "arm64-v8a"):
            assert all(g in args for g in gpu) and "-DGGML_BACKEND_DL=ON" in args
        else:
            assert opencl == ["-DGGML_OPENCL=OFF"] and not any("OPENCL_USE_ADRENO" in a for a in args)
        metal = [a for a in args if a.startswith("-DGGML_METAL")]
        if sdk == "android":
            assert metal == ["-DGGML_METAL=OFF"]
        else:
            assert metal == ["-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON"], (sdk, arch, args)
        assert "-DGGML_VULKAN=OFF" in args                             # Vulkan stays off everywhere


def test_patch_defers_metal_and_preloads_it():
    text = PATCH.read_text(encoding="utf-8")
    added = "\n".join(line[1:] for line in text.splitlines() if line.startswith("+"))
    assert 'if (getenv("GLOSSARION_GGML_EAGER_METAL") != nullptr) {\n            register_backend(ggml_backend_metal_reg());' \
        in added
    assert '"llama": ("ggml-base", "ggml-cpu", "ggml-metal", "ggml"),' in added
    assert "def register_gpu_backend() -> dict:" in added and "ggml_backend_register" in added
    # the description names the new section
    assert "8. Optional GPU backends" in text.split("diff --git", 1)[0]


# ---- the patched loader's register_gpu_backend over a fake ctypes --------------------------------------------

class _Size:
    def __init__(self, value=0):
        self.value = value


def _gpu_ctypes(cdll_ok=(), devices=(), load_ok=True):
    """A fake ctypes + libggml: ``devices`` [(description, total memory)] of the loaded GPU backend."""
    calls = {"load": [], "register": [], "cdll": []}

    def cdll(name, mode=0):
        calls["cdll"].append(name)
        if name.rsplit("/", 1)[-1] not in cdll_ok:
            raise OSError(f'dlopen failed: library "libOpenCL.so" not found: needed by {name}')
        return types.SimpleNamespace()

    def memory(dev, free, total):
        total.value = devices[dev][1]
        free.value = devices[dev][1]

    libggml = types.SimpleNamespace(
        ggml_backend_load=_FakeFn(lambda path: calls["load"].append(path.decode()) or (0x51 if load_ok else None)),
        ggml_backend_register=_FakeFn(lambda reg: calls["register"].append(reg)),
        ggml_backend_reg_dev_count=_FakeFn(lambda reg: len(devices)),
        ggml_backend_reg_dev_get=_FakeFn(lambda reg, i: i),
        ggml_backend_dev_description=_FakeFn(lambda dev: devices[dev][0].encode()),
        ggml_backend_dev_memory=_FakeFn(memory))
    ns = types.SimpleNamespace(CDLL=cdll, RTLD_LOCAL=0, c_int=int, c_size_t=_Size, c_void_p=int, c_char_p=bytes,
                               POINTER=lambda t: t, byref=lambda x: x)
    return ns, libggml, calls


def test_patched_loader_registers_the_opencl_backend_only_on_demand(tmp_path, monkeypatch):
    ns_ctypes, libggml, calls = _gpu_ctypes(devices=[("QUALCOMM Adreno(TM) 830", 11 << 30)])
    ns = _patched_cpu_loader(ns_ctypes)
    assert ns["GPU_STATE"] == {} and ns["gpu_backend_name"]() is None     # nothing loaded yet: no backend known
    (tmp_path / "libggml-opencl.so").write_bytes(b"")
    ns["_LIBGGML"], ns["_BASE_PATH"] = libggml, tmp_path
    assert ns["gpu_backend_name"]() == "OpenCL" and calls["load"] == []    # naming it loads nothing
    state = ns["register_gpu_backend"]()
    assert state == {"backend": "OpenCL", "device": "QUALCOMM Adreno(TM) 830", "error": None}
    assert calls["load"] == [str(tmp_path / "libggml-opencl.so")]
    assert ns["register_gpu_backend"]() == state and len(calls["load"]) == 1   # once per process
    # Android: the libs live in jniLibs (no file next to libggml): bare soname, arm64 only
    ns_ctypes, libggml, calls = _gpu_ctypes(devices=[("QUALCOMM Adreno(TM) 740", 8 << 30)])
    ns = _patched_cpu_loader(ns_ctypes)
    ns["_LIBGGML"], ns["_BASE_PATH"] = libggml, tmp_path / "sitepackages.zip" / "llama_cpp" / "lib"
    monkeypatch.setattr(ns["sys"], "platform", "android")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    assert ns["register_gpu_backend"]()["device"] == "QUALCOMM Adreno(TM) 740"
    assert calls["load"] == ["libggml-opencl.so"]
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    ns = _patched_cpu_loader(_gpu_ctypes()[0])
    ns["_LIBGGML"] = libggml
    assert ns["register_gpu_backend"]() == {"backend": None, "device": None, "error": "this build has no GPU backend"}


def test_patched_loader_explains_why_there_is_no_gpu(tmp_path):
    (tmp_path / "libggml-opencl.so").write_bytes(b"")
    # no libOpenCL.so on the phone: ggml_backend_load fails, the dlopen error says why
    ns_ctypes, libggml, calls = _gpu_ctypes(load_ok=False)
    ns = _patched_cpu_loader(ns_ctypes)
    ns["_LIBGGML"], ns["_BASE_PATH"] = libggml, tmp_path
    state = ns["register_gpu_backend"]()
    assert state["device"] is None and state["error"].startswith('no OpenCL driver (dlopen failed: library "libOpenCL.so"')
    # OpenCL but no Adreno (Mali, Xclipse): the backend drops the device
    ns_ctypes, libggml, _calls = _gpu_ctypes(devices=[])
    ns = _patched_cpu_loader(ns_ctypes)
    ns["_LIBGGML"], ns["_BASE_PATH"] = libggml, tmp_path
    assert ns["register_gpu_backend"]()["error"].startswith("no supported GPU: llama.cpp's OpenCL backend runs on "
                                                             "Qualcomm Adreno")
    # before the engine's libraries loaded
    ns = _patched_cpu_loader(_gpu_ctypes()[0])
    assert ns["register_gpu_backend"]()["error"] == "the engine's libraries are not loaded"


def test_patched_loader_registers_metal_on_demand():
    ns_ctypes, libggml, calls = _gpu_ctypes(devices=[("Apple A17 Pro GPU", 6 << 30)])
    ns = _patched_cpu_loader(ns_ctypes)
    metal = types.SimpleNamespace(ggml_backend_metal_reg=_FakeFn(lambda: 0x77))
    ns["_LIBGGML"], ns["_LIBGGML_METAL"] = libggml, metal
    assert ns["gpu_backend_name"]() == "Metal" and calls["register"] == []
    assert ns["register_gpu_backend"]() == {"backend": "Metal", "device": "Apple A17 Pro GPU", "error": None}
    assert calls["register"] == [0x77]
    # the simulator / a VM: Metal keeps a device without an MTLDevice (no name, no memory)
    ns_ctypes, libggml, _calls = _gpu_ctypes(devices=[("", 0)])
    ns = _patched_cpu_loader(ns_ctypes)
    ns["_LIBGGML"], ns["_LIBGGML_METAL"] = libggml, metal
    assert ns["register_gpu_backend"]()["error"] == "Metal found no usable GPU"


# ---- verify profile ------------------------------------------------------------------------------------------

def test_profile_lists_the_gpu_libraries():
    profile = wh.SITE_PROFILES["llama-cpp-python"]
    assert "libggml-opencl" in wh.profile_libs(profile, "arm64-v8a")
    assert "libggml-opencl" not in wh.profile_libs(profile, "x86_64")
    assert "libggml-metal" not in wh.profile_libs(profile, "x86_64")
    for target in ("iphoneos.arm64", "iphonesimulator.arm64", "iphonesimulator.x86_64"):
        assert wh.profile_libs(profile, target) == wh.IOS_LLAMA_LIBS and "libggml-metal" in wh.IOS_LLAMA_LIBS
    assert profile["android_optional"] == {"libggml-opencl": {"needed": "libOpenCL.so", "imports": r"^cl[A-Z]\w*$"}}


def test_android_gpu_backend_is_accepted_and_kept_optional(tmp_path):
    sp = tmp_path / "ok"
    make_llama_android(sp, "arm64-v8a")
    errors, notes = _verify_llama(sp, "android", make_ndk(tmp_path / "ndk", LLAMA_NDK_EXPORTS))
    assert errors == []
    assert any("lib/libggml-opencl.so optional GPU backend: NEEDED libOpenCL.so" in n for n in notes)
    assert any(re.search(r"lib/libggml-opencl\.so NEEDED .*; \d+ strong imports; [\d.]+ MB", n) for n in notes)
    # libOpenCL.so anywhere else would make the whole engine unloadable on phones without OpenCL
    errors, _ = _verify_llama(make_llama_android(tmp_path / "a", "arm64-v8a", extra_needed=("libOpenCL.so",)).parent,
                              "android", make_ndk(tmp_path / "ndk2", LLAMA_NDK_EXPORTS))
    assert any("lib/libllama.so: NEEDED ['libOpenCL.so'] beyond" in e for e in errors), errors
    # nothing may need the GPU backend itself
    errors, _ = _verify_llama(make_llama_android(tmp_path / "b", "arm64-v8a", extra_needed=("libggml-opencl.so",)).parent,
                              "android", make_ndk(tmp_path / "ndk3", LLAMA_NDK_EXPORTS))
    assert any("NEEDED ['libggml-opencl.so'] beyond" in e and "only loaded on demand" in e for e in errors), errors
    # a GPU backend that does not need libOpenCL.so was built wrong; a non-cl import is still checked
    sp = tmp_path / "c"
    arch = make_llama_android(sp, "arm64-v8a")
    (arch / "llama_cpp" / "lib" / "libggml-opencl.so").write_bytes(make_elf(
        needed=("libggml-base.so", "libc.so"), symbols=[("clFinish", GLOBAL, UND), ("not_there", GLOBAL, UND)],
        code=base.A64_BASE))
    errors, _ = _verify_llama(sp, "android", make_ndk(tmp_path / "ndk4", LLAMA_NDK_EXPORTS))
    text = "\n".join(errors)
    assert "libggml-opencl.so: does not need libOpenCL.so" in text and "1 undefined symbols ['not_there']" in text


def test_ios_metal_backend_may_link_metal_alone(tmp_path):
    for target in base.IOS_SLICES:
        base.make_llama_ios(tmp_path, target)
    errors, notes = _verify_llama(tmp_path, "ios")
    assert errors == [] and any("iphoneos.arm64: lib/libggml-metal.dylib" in n for n in notes)


# ---- link-time OpenCL staging -------------------------------------------------------------------------------

def test_opencl_sources_are_pinned_and_fetched_for_the_android_engine(tmp_path):
    ctx = wh.Ctx(wh.DEFAULT_MANIFEST, "android")
    sources = wh.opencl_sources(ctx)
    assert sorted(sources) == ["headers", "icd_loader"] and wh.opencl_abis(ctx) == ["arm64-v8a"]
    for item in sources.values():
        assert re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) and item["url"].startswith("https://github.com/KhronosGroup/")
    assert wh.needs_opencl(ctx, ["llama-cpp-python"]) and not wh.needs_opencl(ctx, ["cryptography", "pillow"])
    assert not wh.needs_opencl(wh.Ctx(wh.DEFAULT_MANIFEST, "ios"), ["llama-cpp-python"])
    plan = {Path(p).name for _u, p, _s in wh.fetch_plan(ctx, ["llama-cpp-python"], tmp_path)}
    assert {s["file"] for s in sources.values()} <= plan
    plan = {Path(p).name for _u, p, _s in wh.fetch_plan(ctx, ["pillow"], tmp_path)}
    assert not {s["file"] for s in sources.values()} & plan


def test_stage_opencl_checks_the_link_stub(tmp_path):
    stub = tmp_path / "libOpenCL.so"
    stub.write_bytes(make_elf(needed=("libc.so",), soname="libOpenCL.so"))
    wh.check_opencl_stub(stub, "arm64-v8a")
    stub.write_bytes(make_elf(needed=("libc.so",), soname="libOpenCL.so.1"))
    with pytest.raises(wh.WheelsError, match="SONAME 'libOpenCL.so.1', expected 'libOpenCL.so'"):
        wh.check_opencl_stub(stub, "arm64-v8a")
    stub.write_bytes(make_elf(machine=62, needed=("libc.so",), soname="libOpenCL.so"))
    with pytest.raises(wh.WheelsError, match="ELF machine 62 is not arm64-v8a"):
        wh.check_opencl_stub(stub, "arm64-v8a")


def test_stage_opencl_refuses_unverified_sources(tmp_path):
    args = types.SimpleNamespace(packages=["llama-cpp-python"], downloads=str(tmp_path / "dl"), work=str(tmp_path / "w"),
                                 ndk=str(tmp_path / "ndk"))
    with pytest.raises(wh.WheelsError, match="missing or its sha256 differs"):
        wh.cmd_stage_opencl(wh.Ctx(wh.DEFAULT_MANIFEST, "android"), args)
    # not needed (iOS, or no engine in this build): nothing happens
    wh.cmd_stage_opencl(wh.Ctx(wh.DEFAULT_MANIFEST, "ios"), args)
    wh.cmd_stage_opencl(wh.Ctx(wh.DEFAULT_MANIFEST, "android"), types.SimpleNamespace(**{**vars(args),
                                                                                       "packages": ["pillow"]}))


def test_extract_tree_rejects_links(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("OpenCL-Headers-x/CL/cl.h")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(wh.WheelsError, match="unexpected member"):
        wh._extract_tree(archive, tmp_path / "out")


# ---- the APK: the vendor driver is optional and never bundled ---------------------------------------------------

def _verify_apk():
    spec = importlib.util.spec_from_file_location("mobile_verify_apk", MOBILE / "ci" / "verify_apk.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def test_verify_apk_checks_the_optional_opencl_driver():
    va = _verify_apk()
    xml = """N: android=http://schemas.android.com/apk/res/android
  E: manifest (line=2)
    E: application (line=10)
      E: uses-native-library (line=12)
        A: http://schemas.android.com/apk/res/android:name(0x01010003)="libOpenCL.so" (Raw: "libOpenCL.so")
        A: http://schemas.android.com/apk/res/android:required(0x0101028e)=false
"""
    tree = va.parse_xmltree(xml)
    assert va.optional_native_libraries(tree) == {"libOpenCL.so": False}
    names = ["lib/arm64-v8a/libggml-opencl.so", "lib/arm64-v8a/libllama.so"]
    assert va.gpu_problems(tree, names) == []
    assert va.gpu_problems(tree, ["lib/x86_64/libllama.so"]) == []            # no GPU backend: nothing to check
    required = va.parse_xmltree(xml.replace("required(0x0101028e)=false", "required(0x0101028e)=true"))
    assert "required=\"true\"" in va.gpu_problems(required, names)[0]
    missing = va.parse_xmltree(xml.replace("uses-native-library", "uses-library"))
    assert "no <uses-native-library" in va.gpu_problems(missing, names)[0]
    bundled = va.gpu_problems(tree, names + ["lib/arm64-v8a/libOpenCL.so"])
    assert bundled and "bundles ['lib/arm64-v8a/libOpenCL.so']" in bundled[0]


def test_native_extension_manifest_declares_the_optional_driver():
    manifest = (MOBILE / "extensions" / "flet_glossarion_native" / "src" / "flutter" / "flet_glossarion_native" /
                "android" / "src" / "main" / "AndroidManifest.xml").read_text(encoding="utf-8")
    assert re.search(r'<uses-native-library\s+android:name="libOpenCL.so"\s+android:required="false"\s*/>', manifest)
