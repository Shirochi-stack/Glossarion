#!/usr/bin/env python3
"""Helpers for the self-built mobile wheels (cryptography 50.0.2, Pillow 12.3.0) in CI.

Stdlib only, Python 3.11+. Every pinned value comes from ``wheelhouse.toml`` (next to this
file); ``assert-selftest`` also reads the pins in ``src/mobile/pyproject.toml``. The manifest
model is shared with ``tools/check_mobile_wheels.py``.

Used by ``build_wheels.sh`` and ``.github/workflows/mobile-wheels.yml``:

    packages         the packages the manifest builds for a platform
    recipe           the recipe directory of a package (checked against the manifest)
    env              shell exports / GitHub outputs: forge + crossenv revs, python-build release,
                     CPython, ABIs, Rust, NDK; writes the PIP_CONSTRAINT file
    fetch            download + sha256-check the support tarball, BeeWare OpenSSL and the sdists
                     (url + hash from uv.lock) into forge's downloads/ under forge's file names
    stage-openssl    swap the support tree's OpenSSL 3.0.x for BeeWare's static 3.5.9, delete every
                     other static OpenSSL in the support tree (it shadows the staged one on forge's
                     link line) and rebuild forge's openssl dependency wheel (forge's make_dep_wheels)
    collect          copy only the promised wheels from forge's dist/ into <cache>/<package>/
    assemble         copy a platform's cached wheels into the wheelhouse, write SHA256SUMS
    provenance       write provenance.json into the wheelhouse
    verify-native    static checks of every native binary in the wheelhouse (ELF / Mach-O),
                     including its undefined symbols: what the device's dynamic linker must
                     resolve (Android: against the NDK's API-level stub libraries)
    assert-selftest  check the device self-test for the shipped versions: a report (selftest-*.json)
                     or a device log with the smoke suite's GLOSSARION_WHEELS line (logcat)

Exit codes: 0 ok, 1 check or build failure, 2 usage error.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOBILE_DIR = HERE.parent.parent
sys.path.insert(0, str(MOBILE_DIR / "tools"))
import check_mobile_wheels as cmw  # noqa: E402  (stdlib only: manifest model, requirement parsing, targets)

DEFAULT_MANIFEST = HERE / "wheelhouse.toml"
DEFAULT_PYPROJECT = MOBILE_DIR / "pyproject.toml"
USER_AGENT = "glossarion-mobile-wheels/1 (+https://github.com/Shirochi-stack/Glossarion)"
PYTHON_BUILD_URL = "https://github.com/flet-dev/python-build/releases/download/{release}/{file}"
GITHUB_ASSET_URL = "https://github.com/{repo}/releases/download/{tag}/{file}"

PLATFORMS = ("android", "ios")
FORGE_HOST = {"android": "android", "ios": "iOS"}           # forge <host> and setup.sh <platform>
SUPPORT_OS = {"android": "android", "ios": "iOS"}           # directory names inside the support tree
RUST_TARGETS = {
    "arm64-v8a": "aarch64-linux-android",
    "x86_64": "x86_64-linux-android",
    "iphoneos.arm64": "aarch64-apple-ios",
    "iphonesimulator.arm64": "aarch64-apple-ios-sim",
    "iphonesimulator.x86_64": "x86_64-apple-ios",            # Rust has no x86_64-apple-ios-sim
}
# Android: ELF machine per ABI. iOS: (Mach-O cputype, platform) per slice.
ELF_MACHINE = {"arm64-v8a": 183, "x86_64": 62}               # EM_AARCH64, EM_X86_64
CPU_TYPE_ARM64, CPU_TYPE_X86_64 = 0x0100000C, 0x01000007
MACHO_EXPECT = {
    "iphoneos.arm64": (CPU_TYPE_ARM64, 2),                   # PLATFORM_IOS
    "iphonesimulator.arm64": (CPU_TYPE_ARM64, 7),            # PLATFORM_IOSSIMULATOR
    "iphonesimulator.x86_64": (CPU_TYPE_X86_64, 7),
}
MACHO_PLATFORMS = {1: "MACOS", 2: "IOS", 3: "TVOS", 4: "WATCHOS", 7: "IOSSIMULATOR"}
LC_BUILD_VERSION = 0x32
# The older per-OS commands ld writes instead of LC_BUILD_VERSION for deployment targets below
# iOS 12; rustc's default iOS target (10.0) gets LC_VERSION_MIN_IPHONEOS.
LC_VERSION_MIN = {0x24: "MACOSX", 0x25: "IPHONEOS", 0x2F: "TVOS", 0x30: "WATCHOS"}
LC_VERSION_MIN_PLATFORM = {0x24: 1, 0x2F: 3, 0x30: 4}       # IPHONEOS depends on the CPU, see below
# LLVM / rustc never target an arm64 simulator below iOS 14.0 (PyPI's ios_13_0 arm64 simulator
# wheels, Pillow's included, carry minos 14.0 too).
IOS_ARM64_SIMULATOR_FLOOR = (14, 0)
ANDROID_PAGE = 0x4000                                         # Google Play: 16 KB LOAD alignment
ANDROID_FORBIDDEN_NEEDED = re.compile(r"^(libpython3\.so|libcrypto.*|libssl.*|libc\+\+_shared\.so)$")
ANDROID_LIBPYTHON = "libpython3.13.so"
CRYPTOGRAPHY_ANDROID_NEEDED = {ANDROID_LIBPYTHON, "libc.so", "libdl.so", "libm.so", "liblog.so"}
# The NDK sysroot directory of each ABI's stub libraries: sysroot/usr/lib/<triple>/<API level>/.
NDK_TRIPLE = {"arm64-v8a": "aarch64-linux-android", "x86_64": "x86_64-linux-android"}
PILLOW_MODULES = ("_imaging", "_imagingft", "_webp")
PYTHON_FRAMEWORK = "@rpath/Python.framework/Python"
IOS_ALLOWED_DYLIB = re.compile(r"^(@rpath/Python\.framework/Python|/usr/lib/.+|/System/Library/.+)$")
# Static OpenSSL archives (stage-openssl keeps exactly the staged pair in the support tree).
OPENSSL_ARCHIVES = ("libcrypto.a", "libssl.a")
# C names of OpenSSL's API (libcrypto + libssl). A native module that imports one of these is
# missing (part of) its OpenSSL: no wheel we build may take OpenSSL from the device.
OPENSSL_SYMBOL = re.compile(
    r"^(OSSL_|OPENSSL_|OpenSSL_|EVP_|SSL_|SSL3_|TLS_|TLSv1|DTLS_|DTLSv1|ERR_|X509|PEM_|BIO_|BN_|RSA_|DSA_|DH_|"
    r"EC_|ECDSA_|ECDH_|ECX_|ED25519|ED448|X25519|X448|CRYPTO_|RAND_|ASN1_|OBJ_|PKCS\d|PKCS_|HMAC|CMAC_|"
    r"d2i_|i2d_|OCSP_|CMS_|CONF_|NCONF_|ENGINE_)")
PYTHON_SYMBOL = re.compile(r"^_?Py")           # the C API (PyObject_*, _Py_*, PyExc_*, ...)
OPENSSL_VERSION_TEXT = re.compile(rb"OpenSSL (\d+\.\d+\.\d+)(?![0-9])")
# ELF symbol bindings / section types.
STB_LOCAL, STB_GLOBAL, STB_WEAK, STB_GNU_UNIQUE = 0, 1, 2, 10
SHT_DYNSYM = 11
# Mach-O: header flag, LC_SYMTAB, nlist types and the special library ordinals of n_desc.
MH_TWOLEVEL = 0x80
LC_SYMTAB = 0x2
N_STAB, N_TYPE, N_EXT, N_UNDF, N_PBUD = 0xE0, 0x0E, 0x01, 0x0, 0xC
MACHO_SPECIAL_ORDINALS = {0x00: "SELF_LIBRARY_ORDINAL", 0xFE: "DYNAMIC_LOOKUP_ORDINAL", 0xFF: "EXECUTABLE_ORDINAL"}
PILLOW_FEATURES = ("jpg", "zlib", "webp", "freetype2")
PILLOW_ROUNDTRIP = ("JPEG", "PNG", "WEBP")
# The smoke self-test (app/glossarion_mobile/diagnostics/selftest.py) prints the fernet and pillow
# checks on one device-log line: GLOSSARION_WHEELS {"suite":"smoke",...,"checks":[...]}.
WHEELS_MARKER = "GLOSSARION_WHEELS"
WHEELS_LINE = re.compile(r"(?:^|[^A-Z_])" + WHEELS_MARKER + r" (\{.*)$")


class WheelsError(RuntimeError):
    """A build or check failure (exit code 1)."""


def log(msg: str) -> None:
    print(f"[wheels] {msg}", flush=True)


# ============================================================================ manifest helpers
class Ctx:
    def __init__(self, manifest_path, platform: str | None, pyproject=None):
        self.manifest_path = Path(manifest_path)
        self.manifest = cmw.load_wheelhouse_manifest(self.manifest_path)
        self.platform = platform
        self.pyproject = Path(pyproject) if pyproject else DEFAULT_PYPROJECT

    @property
    def data(self) -> dict:
        return self.manifest.data

    @property
    def toolchain(self) -> dict:
        return self.data.get("toolchain") or {}

    @property
    def py_full(self) -> str:
        return str(self.toolchain["python_version"])

    @property
    def py_short(self) -> str:
        return ".".join(self.py_full.split(".")[:2])

    @property
    def ios_deployment_target_text(self) -> str:
        return str(self.toolchain.get("ios_deployment_target", "13.0"))

    @property
    def ios_deployment_target(self) -> tuple:
        return tuple(int(x) for x in self.ios_deployment_target_text.split(".")[:2])

    def wheels(self, packages=None) -> list:
        """The promised wheels with targets on this platform (all, or the named packages)."""
        on = [w for w in self.manifest.wheels if any(cmw.on_platform(t, self.platform) for t in w.targets)]
        if not packages:
            return on
        wanted = [cmw.canonical(p) for p in packages]
        unknown = [p for p in wanted if p not in {w.name for w in on}]
        if unknown:
            raise WheelsError(f"{', '.join(unknown)}: not built for {self.platform} by {self.manifest_path.name} "
                              f"(it builds: {' '.join(w.name for w in on) or 'nothing'})")
        return [w for w in on if w.name in wanted]

    def targets(self, packages=None) -> list:
        """Target keys of this platform the wheels need, in TARGETS order."""
        need = {t for w in self.wheels(packages) for t in w.targets if cmw.on_platform(t, self.platform)}
        return [t for t in cmw.TARGETS if t in need]

    def recipe_dir(self, wheel) -> Path:
        path = (self.manifest_path.parent / wheel.recipe).resolve()
        meta = recipe_meta(path / "meta.yaml")
        if cmw.parse_version(meta.get("version", "")) != cmw.parse_version(wheel.version):
            raise WheelsError(f"{path}/meta.yaml builds version {meta.get('version')!r}, "
                              f"{self.manifest_path.name} promises {wheel.name} {wheel.version}")
        if meta.get("number") != str(self.manifest.build_tag):
            raise WheelsError(f"{path}/meta.yaml has build number {meta.get('number')!r}, "
                              f"{self.manifest_path.name} has build_tag {self.manifest.build_tag}")
        if cmw.canonical(meta.get("name", "")) != wheel.name:
            raise WheelsError(f"{path}/meta.yaml builds {meta.get('name')!r}, not {wheel.name}")
        if any(cmw.platform_of(t) == "ios" for t in wheel.targets) and \
                meta.get("ios_deployment_target") != self.ios_deployment_target_text:
            # forge's compile_env() starts from an empty environment, so an exported
            # IPHONEOS_DEPLOYMENT_TARGET never reaches rustc and the slices would need iOS 10.0.
            raise WheelsError(f"{path}/meta.yaml: build.script_env sets IPHONEOS_DEPLOYMENT_TARGET "
                              f"{meta.get('ios_deployment_target')!r}, {self.manifest_path.name} has "
                              f"ios_deployment_target {self.ios_deployment_target_text!r} (set it in the "
                              f"recipe's iOS branch: forge passes no outer environment to the build)")
        return path

    def needs_openssl(self, packages=None) -> bool:
        return any(recipe_meta(self.recipe_dir(w) / "meta.yaml").get("openssl") for w in self.wheels(packages))


def recipe_meta(path: Path) -> dict:
    """package.name, package.version, build.number, whether requirements.host lists openssl and
    build.script_env's IPHONEOS_DEPLOYMENT_TARGET (``ios_deployment_target``).

    A line-based read of our own forge recipes (no YAML parser in the stdlib). Lines that are
    comments (including the ``# {% if %}`` Jinja branches) are skipped, so a value set in any
    branch counts.
    """
    out, section, sub = {}, "", ""
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            section, sub = line.split(":", 1)[0].strip(), ""
            continue
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if indent == 2 and stripped.endswith(":"):
            sub = stripped[:-1]
            continue
        m = re.match(r"^(\w+):\s*['\"]?([^'\"#\s]+)['\"]?", stripped)
        if section == "package" and indent == 2 and m and m.group(1) in ("name", "version"):
            out[m.group(1)] = m.group(2)
        elif section == "build" and indent == 2 and m and m.group(1) == "number":
            out["number"] = m.group(2)
        elif section == "build" and sub == "script_env" and indent == 4 and m \
                and m.group(1) == "IPHONEOS_DEPLOYMENT_TARGET":
            out["ios_deployment_target"] = m.group(2)
        elif section == "requirements" and sub == "host" and re.match(r"^-\s*openssl(\s|$)", stripped):
            out["openssl"] = stripped[1:].strip()
    return out


file_sha256 = cmw.file_sha256


def pinned_version(pyproject: Path, name: str) -> str | None:
    """The ``==`` pin of ``name`` in pyproject (dependencies or [tool.flet.<os>].dependencies)."""
    project = cmw.load_project(pyproject)
    for req in project.common + project.android + project.ios:
        if req.key == cmw.canonical(name):
            specs = req.spec.specs
            if len(specs) == 1 and specs[0].op == "==" and not specs[0].wild:
                return specs[0].ver
    return None


def openssl_spec(ctx: Ctx) -> tuple:
    """(version, platform table) of [openssl]."""
    o = ctx.data.get("openssl") or {}
    spec = o.get(ctx.platform)
    if not o.get("version") or not isinstance(spec, dict):
        raise WheelsError(f"{ctx.manifest_path.name}: no [openssl] / [openssl.{ctx.platform}]")
    return str(o["version"]), spec


def openssl_targets(ctx: Ctx) -> list:
    """The targets forge's make_dep_wheels walks: ANDROID_ABIS (setup.sh exports it) or all iOS slices."""
    if ctx.platform == "android":
        abis = os.environ.get("ANDROID_ABIS", "").split()
        return abis or ctx.targets()
    return list(cmw.IOS_DEFAULT_ARCHS)


# ============================================================================ downloads
def download_verified(url: str, dest: Path, sha256: str, attempts: int = 4) -> None:
    """Download ``url`` to ``dest`` unless an identical file is there; fail unless sha256 matches."""
    dest = Path(dest)
    if dest.is_file():
        if file_sha256(dest) == sha256:
            log(f"ok (already present) {dest.name}")
            return
        log(f"{dest.name}: present but its sha256 differs; downloading again")
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h, last = hashlib.sha256(), None
    for attempt in range(attempts):
        h, last = hashlib.sha256(), None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as out:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
                    out.write(chunk)
            break
        except urllib.error.HTTPError as e:
            last = e
            if 400 <= e.code < 500 and e.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        if attempt + 1 < attempts:
            time.sleep(2 * (attempt + 1))
    if last is not None:
        part.unlink(missing_ok=True)
        raise WheelsError(f"GET {url} failed: {last}")
    got = h.hexdigest()
    if got != sha256:
        part.unlink(missing_ok=True)
        raise WheelsError(f"{url}: sha256 {got} does not match the pinned {sha256}")
    os.replace(part, dest)
    log(f"ok {dest.name} sha256 {got}")


def fetch_plan(ctx: Ctx, packages, dest: Path) -> list:
    """[(url, path, sha256)] for the support tarball, OpenSSL (when a recipe needs it) and the sdists."""
    tc = ctx.toolchain
    tarball = f"python-{ctx.platform}-mobile-forge-{ctx.py_full}.tar.gz"
    support_sha = (tc.get("support_sha256") or {}).get(ctx.platform)
    if not support_sha:
        raise WheelsError(f"{ctx.manifest_path.name}: no toolchain.support_sha256.{ctx.platform}")
    items = [(PYTHON_BUILD_URL.format(release=tc["python_build_release"], file=tarball), dest / tarball, support_sha)]
    if ctx.needs_openssl(packages):
        _version, spec = openssl_spec(ctx)
        for target in openssl_targets(ctx):
            asset = (spec.get("assets") or {}).get(target)
            if not asset:
                raise WheelsError(f"{ctx.manifest_path.name}: no OpenSSL asset for {target}")
            url = GITHUB_ASSET_URL.format(repo=spec["repo"], tag=spec["tag"], file=asset["file"])
            items.append((url, dest / asset["file"], asset["sha256"]))
    lock_path = ctx.pyproject.parent / "uv.lock"
    lock = cmw.load_uv_lock_sdists(lock_path)
    for w in ctx.wheels(packages):
        entry = lock.get((w.name, w.version))
        if entry is None or not entry[0]:
            raise WheelsError(f"{lock_path}: no sdist for {w.name} {w.version}")
        url, locked = entry
        if locked != w.sdist_sha256:
            raise WheelsError(f"{w.name} {w.version}: uv.lock sdist sha256 {locked} != wheelhouse.toml {w.sdist_sha256}")
        # forge's PythonPackageBuilder reads downloads/<recipe package.name>-<version>.tar.gz
        forge_name = recipe_meta(ctx.recipe_dir(w) / "meta.yaml")["name"]
        items.append((url, dest / f"{forge_name}-{w.version}.tar.gz", w.sdist_sha256))
    return items


# ============================================================================ OpenSSL staging
def _tar_rel(name: str) -> str:
    return name[2:] if name.startswith("./") else name


def extract_static_openssl(tarball: Path, dest: Path, version: str) -> None:
    """Unpack only include/ and lib/lib{crypto,ssl}.a, so openssl-sys can only link statically."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    keep_libs = {"lib/libcrypto.a", "lib/libssl.a"}
    with tarfile.open(tarball, "r:gz") as tf:
        members = []
        for m in tf.getmembers():
            rel = _tar_rel(m.name)
            if not (rel.startswith("include/") or rel in keep_libs):
                continue
            if cmw.unsafe_member_name(rel):
                raise WheelsError(f"{tarball.name}: unsafe member {m.name}")
            if m.isfile() or m.isdir():
                members.append(m)
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tf.extractall(dest, members=members, **kwargs)
    missing = [p for p in sorted(keep_libs) if not (dest / p).is_file()]
    header = dest / "include" / "openssl" / "opensslv.h"
    if missing or not header.is_file():
        raise WheelsError(f"{tarball.name}: missing {missing or [str(header.relative_to(dest))]}")
    if f'OPENSSL_VERSION_STR "{version}"' not in re.sub(r"\s+", " ", header.read_text(encoding="utf-8")):
        raise WheelsError(f"{tarball.name}: opensslv.h is not OpenSSL {version}")
    files = [p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file() or p.is_symlink()]
    leftovers = sorted(f for f in files if not f.startswith("include/") and f not in keep_libs)
    if leftovers:
        raise WheelsError(f"{tarball.name}: unexpected files after extraction: {leftovers[:5]}")


def rewrite_versions(versions: Path, version: str, build: str) -> None:
    """Point the support tree's VERSIONS openssl line (``openssl:`` / ``OpenSSL:``) at version-build."""
    text = versions.read_text(encoding="utf-8")
    new, n = re.subn(r"(?im)^(openssl):[ \t]*\S+[ \t]*$", lambda m: f"{m.group(1)}: {version}-{build}", text)
    if n != 1:
        raise WheelsError(f"{versions}: expected one openssl line, found {n}")
    versions.write_text(new, encoding="utf-8")


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def shadow_openssl_archives(support: Path, staged) -> list:
    """Every lib{crypto,ssl}.a in the support tree outside the ``staged`` OpenSSL directories."""
    keep = [Path(d).resolve() for d in staged]
    found = []
    for path in sorted(Path(support).rglob("lib*.a")):
        if path.name in OPENSSL_ARCHIVES and not any(_is_within(path.parent.resolve() / path.name, k) for k in keep):
            found.append(path)
    return found


def remove_shadow_openssl(support: Path, staged) -> list:
    """Delete the static OpenSSL archives the support tree ships outside the staged directories.

    forge links Rust with ``CARGO_TARGET_<triple>_RUSTFLAGS=" -L{prefix}/lib ..."`` and cargo puts
    RUSTFLAGS before the build scripts' ``-L native=$OPENSSL_DIR/lib``, so openssl-sys's
    ``-l static=crypto`` / ``static=ssl`` take the first libcrypto.a / libssl.a on that path. On
    Android {prefix}/lib is install/android/<abi>/python-3.13.x/lib, which holds CPython's own
    static OpenSSL 3.0.x: cryptography got compiled against the staged 3.5.9 headers but linked
    3.0.21, the 3.2+ symbols (OSSL_get_max_threads, ...) stayed undefined and dlopen failed on the
    device. The old openssl-3.0.x-*/lib archives go too, so exactly the staged pair is left. Shared
    libraries (libcrypto_python.so for CPython's _ssl / _hashlib) are kept.
    """
    removed = []
    for path in shadow_openssl_archives(support, staged):
        path.unlink()
        removed.append(path)
    return removed


def check_openssl_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
    payload = [n for n in names if ".dist-info/" not in n]
    bad = [n for n in payload if not (n.startswith("opt/include/") or n in ("opt/lib/libcrypto.a", "opt/lib/libssl.a"))]
    if bad or "opt/lib/libcrypto.a" not in payload or "opt/lib/libssl.a" not in payload:
        raise WheelsError(f"{path.name}: unexpected contents {bad[:5] or payload[:5]}")


def cmd_stage_openssl(ctx: Ctx, args) -> None:
    if not ctx.needs_openssl(args.packages):
        log("no recipe of these packages needs openssl; nothing to stage")
        return
    version, spec = openssl_spec(ctx)
    build = str(spec["build"])
    env_var = "MOBILE_FORGE_ANDROID_SUPPORT_PATH" if ctx.platform == "android" else "MOBILE_FORGE_IOS_SUPPORT_PATH"
    support = Path(os.environ.get(env_var) or "")
    if not str(support) or not (support / "support").is_dir():
        raise WheelsError(f"{env_var} does not point at an extracted support tree (source setup.sh first)")
    os_name = SUPPORT_OS[ctx.platform]
    downloads = Path(args.downloads)
    staged = []
    for target in openssl_targets(ctx):
        asset = spec["assets"][target]
        tarball = downloads / asset["file"]
        if not tarball.is_file() or file_sha256(tarball) != asset["sha256"]:
            raise WheelsError(f"{tarball}: missing or its sha256 differs from the manifest (run fetch)")
        dest = support / "install" / os_name / target / f"openssl-{version}-{build}"
        extract_static_openssl(tarball, dest, version)
        staged.append(dest)
        log(f"staged static OpenSSL {version} for {target} in {dest}")
    for path in remove_shadow_openssl(support, staged):
        log(f"removed {path.relative_to(support).as_posix()}: a static OpenSSL other than the staged {version} "
            f"(forge's cargo -L {{prefix}}/lib comes before OPENSSL_DIR/lib, so it would be linked instead)")
    left = shadow_openssl_archives(support, staged)
    if left:
        raise WheelsError(f"static OpenSSL archives other than the staged {version} remain in {support}: "
                          f"{[p.relative_to(support).as_posix() for p in left]}")
    rewrite_versions(support / "support" / ctx.py_short / os_name / "VERSIONS", version, build)
    forge = Path(args.forge)
    dist = forge / "dist"
    for old in sorted(dist.glob("openssl-*.whl")):
        log(f"removing {old.name}")
        old.unlink()
    subprocess.run([sys.executable, "-m", "make_dep_wheels", os_name], cwd=forge, check=True)
    for target in openssl_targets(ctx):
        whl = dist / f"openssl-{version}-{build}-py3-none-{cmw.TARGETS[target].label}.whl"
        if not whl.is_file():
            raise WheelsError(f"make_dep_wheels did not produce {whl.name}")
        check_openssl_wheel(whl)
        log(f"ok {whl.name}")


# ============================================================================ collect / assemble / provenance
def _wheel_identity(filename: str):
    parts = filename[:-4].split("-") if filename.endswith(".whl") else []
    if len(parts) != 6:
        return None
    name, ver, build, py, abi, plat = parts
    v = cmw.parse_version(ver)
    if v is None:
        return None
    return cmw.canonical(name), v, build, (py, abi, plat)


def find_wheel(directory: Path, expected: str) -> list:
    want = _wheel_identity(expected)
    return sorted(p for p in Path(directory).glob("*.whl") if _wheel_identity(p.name) == want)


def _run_text(cmd) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout or out.stderr or "").strip()
    return " ".join(text.split()) or None


def toolchain_info(ctx: Ctx, logs: Path | None) -> dict:
    tc = ctx.toolchain
    info = {
        "mobile_forge": (tc.get("mobile_forge") or {}).get("rev"),
        "crossenv": (tc.get("crossenv") or {}).get("rev"),
        "python_build_release": tc.get("python_build_release"),
        "python_version": tc.get("python_version"),
        "image": " ".join(x for x in (os.environ.get("ImageOS"), os.environ.get("ImageVersion")) if x) or None,
        "runner": " ".join(x for x in (os.environ.get("RUNNER_OS"), os.environ.get("RUNNER_ARCH")) if x) or None,
        "rustc": _run_text(["rustc", "-V"]),
    }
    if ctx.platform == "android":
        ndk = os.environ.get("NDK_HOME")
        props = Path(ndk) / "source.properties" if ndk else None
        if props is not None and props.is_file():
            m = re.search(r"Pkg\.Revision\s*=\s*(\S+)", props.read_text(encoding="utf-8", errors="replace"))
            info["ndk"] = m.group(1) if m else ndk
        else:
            info["ndk"] = ndk
    else:
        info["xcode"] = _run_text(["xcodebuild", "-version"])
    found = set()
    if logs is not None and Path(logs).is_dir():
        for p in sorted(Path(logs).glob("*.log")):
            text = p.read_text(encoding="utf-8", errors="replace")
            found.update(re.findall(r"\bmaturin-(\d+\.\d+\.\d+)\b", text))
    info["maturin"] = sorted(found) or None
    return info


def check_openssl_source(log_text: str, version: str, build: str, label: str) -> str | None:
    """None when forge's pip took ``openssl==<version>`` from dist/ (the staged static OpenSSL).

    forge installs host requirements with ``--find-links dist`` next to PyPI and pypi.flet.dev, and
    no project named ``openssl`` exists on PyPI today; a later upload there could otherwise win.
    """
    if re.search(r"Downloading\s+\S*openssl-", log_text, re.IGNORECASE):
        return "pip downloaded an openssl distribution from an index instead of using the staged dist/ wheel"
    wanted = rf"Processing\s+\S*/dist/openssl-{re.escape(version)}-{re.escape(build)}-py3-none-{re.escape(label)}\.whl"
    if not re.search(wanted, log_text):
        return f"the forge log does not show pip installing dist/openssl-{version}-{build}-py3-none-{label}.whl"
    return None


def cmd_collect(ctx: Ctx, args) -> None:
    dist, cache = Path(args.dist), Path(args.cache)
    info = toolchain_info(ctx, Path(args.logs) if args.logs else None)
    for w in ctx.wheels(args.packages):
        if args.logs and ctx.needs_openssl([w.name]):
            version, spec = openssl_spec(ctx)
            forge_name = recipe_meta(ctx.recipe_dir(w) / "meta.yaml")["name"]
            for target in w.targets:
                if not cmw.on_platform(target, ctx.platform):
                    continue
                label = cmw.TARGETS[target].label
                log_path = Path(args.logs) / f"{forge_name}-{w.version}-cp{ctx.py_short.replace('.', '')}-{label}.log"
                text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
                problem = check_openssl_source(text, version, str(spec["build"]), label) if text else \
                    f"no forge log {log_path.name}"
                if problem:
                    raise WheelsError(f"{w.name} for {target}: {problem}")
                log(f"{w.name} for {target}: built against dist/openssl-{version}-{spec['build']} (static)")
        out_dir = cache / w.name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        hashes = {}
        for target in w.targets:
            if not cmw.on_platform(target, ctx.platform):
                continue
            expected = w.filename(target, ctx.manifest.build_tag)
            matches = find_wheel(dist, expected)
            if len(matches) != 1:
                raise WheelsError(f"{dist}: expected one {expected}, found {[p.name for p in matches]}")
            shutil.copy2(matches[0], out_dir / matches[0].name)
            hashes[matches[0].name] = file_sha256(out_dir / matches[0].name)
            log(f"collected {matches[0].name}")
        build_info = {
            "package": w.name, "version": w.version, "platform": ctx.platform,
            "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "workflow_run": _run_env(), "toolchain": info, "sha256": hashes,
        }
        (out_dir / "build-info.json").write_text(json.dumps(build_info, indent=1) + "\n", encoding="utf-8")


def _run_env() -> dict:
    keys = ("GITHUB_REPOSITORY", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA", "GITHUB_REF", "GITHUB_WORKFLOW",
            "GITHUB_EVENT_NAME", "GITHUB_ACTOR")
    return {k.lower(): os.environ.get(k) for k in keys if os.environ.get(k)}


def cmd_assemble(ctx: Ctx, args) -> None:
    cache, out = Path(args.cache), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows = []
    for w, target in ctx.manifest.promised(ctx.platform):
        expected = w.filename(target, ctx.manifest.build_tag)
        matches = find_wheel(cache / w.name, expected)
        if len(matches) != 1:
            raise WheelsError(f"{cache / w.name}: expected one {expected}, found {[p.name for p in matches]}")
        shutil.copy2(matches[0], out / matches[0].name)
    lines = []
    for whl in sorted(out.glob("*.whl")):
        digest = file_sha256(whl)
        lines.append(f"{digest}  {whl.name}")
        rows.append(f"| `{whl.name}` | {whl.stat().st_size:,} | `{digest}` |")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wheelhouse {out}: {len(lines)} wheels")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"## Mobile wheelhouse ({ctx.platform})\n\n| wheel | bytes | sha256 |\n|---|---|---|\n")
            f.write("\n".join(rows) + "\n\n")


def cmd_provenance(ctx: Ctx, args) -> None:
    wheelhouse, cache = Path(args.wheelhouse), Path(args.cache) if args.cache else None
    built = {cmw.canonical(p) for p in (args.built or "").split()}
    version, spec = openssl_spec(ctx)
    entries = []
    for whl in sorted(wheelhouse.glob("*.whl")):
        ident = _wheel_identity(whl.name)
        pkg = ident[0] if ident else ""
        info_path = cache / pkg / "build-info.json" if cache is not None and pkg else None
        entries.append({
            "file": whl.name, "sha256": file_sha256(whl), "size": whl.stat().st_size, "package": pkg,
            "built_in_this_run": pkg in built,
            "build": json.loads(info_path.read_text(encoding="utf-8")) if info_path and info_path.is_file() else None,
        })
    doc = {
        "schema": 1,
        "platform": ctx.platform,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest": {"path": "src/mobile/ci/wheels/wheelhouse.toml", "sha256": file_sha256(ctx.manifest_path),
                     "build_tag": ctx.manifest.build_tag, "toolchain": ctx.toolchain,
                     "openssl": {"version": version, **spec}},
        "workflow_run": _run_env(),
        "image": " ".join(x for x in (os.environ.get("ImageOS"), os.environ.get("ImageVersion")) if x) or None,
        "wheels": entries,
    }
    (wheelhouse / "provenance.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    log(f"wrote {wheelhouse / 'provenance.json'} ({len(entries)} wheels)")


# ============================================================================ native binaries
def parse_elf(data: bytes) -> dict:
    """machine, class, LOAD alignments and DT_NEEDED of an ELF shared object."""
    if data[:4] != b"\x7fELF":
        raise ValueError("not an ELF file")
    ei_class, ei_data = data[4], data[5]
    end = "<" if ei_data == 1 else ">"
    if ei_class == 2:
        (e_machine,) = struct.unpack_from(end + "H", data, 18)
        (e_phoff,) = struct.unpack_from(end + "Q", data, 32)
        e_phentsize, e_phnum = struct.unpack_from(end + "HH", data, 54)
        ph_fmt, dyn_fmt = end + "IIQQQQQQ", end + "qQ"
    elif ei_class == 1:
        (e_machine,) = struct.unpack_from(end + "H", data, 18)
        (e_phoff,) = struct.unpack_from(end + "I", data, 28)
        e_phentsize, e_phnum = struct.unpack_from(end + "HH", data, 42)
        ph_fmt, dyn_fmt = end + "IIIIIIII", end + "iI"
    else:
        raise ValueError(f"unknown ELF class {ei_class}")
    loads, dynamic = [], None
    for i in range(e_phnum):
        fields = struct.unpack_from(ph_fmt, data, e_phoff + i * e_phentsize)
        if ei_class == 2:
            p_type, _flags, p_offset, p_vaddr, _paddr, p_filesz, _memsz, p_align = fields
        else:
            p_type, p_offset, p_vaddr, _paddr, p_filesz, _memsz, _flags, p_align = fields
        if p_type == 1:      # PT_LOAD
            loads.append((p_vaddr, p_offset, p_filesz, p_align))
        elif p_type == 2:    # PT_DYNAMIC
            dynamic = (p_offset, p_filesz)

    def vaddr_to_offset(addr: int) -> int:
        for vaddr, offset, filesz, _align in loads:
            if vaddr <= addr < vaddr + filesz:
                return offset + (addr - vaddr)
        raise ValueError(f"address {addr:#x} is outside every LOAD segment")

    needed_offsets, strtab = [], None
    if dynamic is not None:
        size = struct.calcsize(dyn_fmt)
        for i in range(dynamic[1] // size):
            tag, val = struct.unpack_from(dyn_fmt, data, dynamic[0] + i * size)
            if tag == 0:
                break
            if tag == 1:     # DT_NEEDED
                needed_offsets.append(val)
            elif tag == 5:   # DT_STRTAB
                strtab = val
    needed = []
    if needed_offsets:
        if strtab is None:
            raise ValueError("DT_NEEDED without DT_STRTAB")
        base = vaddr_to_offset(strtab)
        for off in needed_offsets:
            endpos = data.index(b"\0", base + off)
            needed.append(data[base + off:endpos].decode("utf-8", "replace"))
    return {"class": 64 if ei_class == 2 else 32, "machine": e_machine,
            "load_aligns": [a for _v, _o, _s, a in loads], "needed": needed}


def elf_dynsym(data: bytes) -> list:
    """[(name, binding, defined)] of the ELF's .dynsym, the table the dynamic linker binds.

    Read through the section headers (SHT_DYNSYM and its sh_link string table). Raises ValueError
    when there is none, so the undefined-symbol checks fail closed on an unreadable file.
    """
    if data[:4] != b"\x7fELF":
        raise ValueError("not an ELF file")
    ei_class, ei_data = data[4], data[5]
    end = "<" if ei_data == 1 else ">"
    if ei_class == 2:
        (e_shoff,) = struct.unpack_from(end + "Q", data, 40)
        e_shentsize, e_shnum = struct.unpack_from(end + "HH", data, 58)
        sh_fmt, sym_fmt = end + "IIQQQQIIQQ", end + "IBBHQQ"
    elif ei_class == 1:
        (e_shoff,) = struct.unpack_from(end + "I", data, 32)
        e_shentsize, e_shnum = struct.unpack_from(end + "HH", data, 46)
        sh_fmt, sym_fmt = end + "IIIIIIIIII", end + "IIIBBH"
    else:
        raise ValueError(f"unknown ELF class {ei_class}")
    if not e_shoff or not e_shnum:
        raise ValueError("no section headers, so no .dynsym to read")
    sections = []   # (type, offset, size, link, entsize); 32- and 64-bit headers share the field order
    for i in range(e_shnum):
        f = struct.unpack_from(sh_fmt, data, e_shoff + i * e_shentsize)
        sections.append((f[1], f[4], f[5], f[6], f[9]))
    dynsyms = [s for s in sections if s[0] == SHT_DYNSYM]
    if len(dynsyms) != 1:
        raise ValueError(f"{len(dynsyms)} SHT_DYNSYM sections (expected one)")
    _type, sym_off, sym_size, link, entsize = dynsyms[0]
    if link >= len(sections):
        raise ValueError(".dynsym names no string table")
    _st, str_off, str_size, _l, _e = sections[link]
    entsize = entsize or struct.calcsize(sym_fmt)
    if sym_off + sym_size > len(data) or str_off + str_size > len(data) or entsize < struct.calcsize(sym_fmt):
        raise ValueError(".dynsym or its string table lies outside the file")
    out = []
    for i in range(sym_size // entsize):
        f = struct.unpack_from(sym_fmt, data, sym_off + i * entsize)
        if ei_class == 2:
            st_name, st_info, _other, st_shndx = f[0], f[1], f[2], f[3]
        else:
            st_name, st_info, _other, st_shndx = f[0], f[3], f[4], f[5]
        if not st_name:
            continue
        if st_name >= str_size:
            raise ValueError(f"symbol name offset {st_name} is outside .dynstr")
        endpos = data.index(b"\0", str_off + st_name)
        out.append((data[str_off + st_name:endpos].decode("utf-8", "replace"), st_info >> 4, st_shndx != 0))
    return out


def default_ndk(ctx: Ctx) -> Path | None:
    """$NDK_HOME, else the manifest's NDK under $ANDROID_HOME / $ANDROID_SDK_ROOT (CI runners have it)."""
    if os.environ.get("NDK_HOME"):
        return Path(os.environ["NDK_HOME"])
    version = ctx.toolchain.get("android_ndk")
    for var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        root = os.environ.get(var)
        if root and version and (Path(root) / "ndk" / str(version)).is_dir():
            return Path(root) / "ndk" / str(version)
    return None


class NdkStubs:
    """Exported symbols of the NDK's stub libraries (sysroot/usr/lib/<triple>/<API>/lib*.so).

    The stubs list what bionic exports at the wheel's API level (24): an import none of a module's
    NEEDED system libraries exports there makes dlopen fail (the module is linked BIND_NOW).
    """

    def __init__(self, ndk):
        self.ndk = Path(ndk) if ndk else None
        self._exports = {}

    def lib_dir(self, target: str) -> tuple:
        """(stub directory or None, why not)."""
        if self.ndk is None:
            return None, "no NDK (pass --ndk, or set NDK_HOME or ANDROID_HOME)"
        triple, api = NDK_TRIPLE.get(target), cmw.TARGETS[target].max_version[0]
        if triple is None:
            return None, f"no NDK triple known for {target}"
        dirs = sorted(self.ndk.glob(f"toolchains/llvm/prebuilt/*/sysroot/usr/lib/{triple}/{api}"))
        if not dirs:
            return None, f"no sysroot/usr/lib/{triple}/{api} stub libraries under {self.ndk}"
        return dirs[0], None

    def exports(self, target: str, libs) -> tuple:
        """(the symbols the libraries ``libs`` export at the target's API level, or None; why not)."""
        directory, problem = self.lib_dir(target)
        if directory is None:
            return None, problem
        out = set()
        for lib in libs:
            path = directory / lib
            if path not in self._exports:
                if not path.is_file():
                    return None, f"{path} not found"
                self._exports[path] = {n for n, bind, defined in elf_dynsym(path.read_bytes())
                                       if defined and bind in (STB_GLOBAL, STB_WEAK, STB_GNU_UNIQUE)}
            out |= self._exports[path]
        return out, None


def parse_macho(data: bytes) -> dict:
    """cputype, header flags, LC_BUILD_VERSION / LC_VERSION_MIN_* versions, dylib dependencies and
    undefined external symbols (with their two-level library ordinal) of a thin 64-bit Mach-O.

    ``dylibs`` are in load-command order, which is the order library ordinals count (1 = first).
    ``undefined`` is None without an LC_SYMTAB.
    """
    if len(data) < 32:
        raise ValueError("too short for Mach-O")
    (magic_be,) = struct.unpack_from(">I", data, 0)
    if magic_be in (0xCAFEBABE, 0xCAFEBABF):
        raise ValueError("universal (fat) binary; each wheel must hold one slice")
    (magic,) = struct.unpack_from("<I", data, 0)
    if magic != 0xFEEDFACF:
        raise ValueError("not a 64-bit little-endian Mach-O file")
    _magic, cputype, _sub, filetype, ncmds, _sizeofcmds, flags, _res = struct.unpack_from("<IiiIIIII", data, 0)
    off = 32
    dylibs, builds, version_mins, symtab = [], [], [], None
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", data, off)
        if cmdsize < 8:
            raise ValueError("corrupt load command")
        if cmd in (0xC, 0x80000018, 0x8000001F, 0x20, 0x80000023):   # LOAD / WEAK / REEXPORT / LAZY / UPWARD_DYLIB
            (name_off,) = struct.unpack_from("<I", data, off + 8)
            raw = data[off + name_off:off + cmdsize]
            dylibs.append(raw.split(b"\0", 1)[0].decode("utf-8", "replace"))
        elif cmd == LC_BUILD_VERSION:
            platform, minos = struct.unpack_from("<II", data, off + 8)
            builds.append((platform, _macho_version(minos)))
        elif cmd in LC_VERSION_MIN:
            (v,) = struct.unpack_from("<I", data, off + 8)
            version_mins.append((cmd, _macho_version(v)))
        elif cmd == LC_SYMTAB:
            symtab = struct.unpack_from("<IIII", data, off + 8)      # symoff, nsyms, stroff, strsize
        off += cmdsize
    undefined = None
    if symtab is not None:
        symoff, nsyms, stroff, strsize = symtab
        if symoff + nsyms * 16 > len(data) or stroff + strsize > len(data):
            raise ValueError("LC_SYMTAB points outside the file")
        strings = data[stroff:stroff + strsize]
        undefined = []
        for n_strx, n_type, _sect, n_desc, _value in struct.iter_unpack("<IBBHQ", data[symoff:symoff + nsyms * 16]):
            if n_type & N_STAB or not n_type & N_EXT or (n_type & N_TYPE) not in (N_UNDF, N_PBUD):
                continue
            if n_strx >= strsize:
                raise ValueError(f"symbol name offset {n_strx} is outside the string table")
            name = strings[n_strx:strings.index(b"\0", n_strx)].decode("utf-8", "replace")
            undefined.append((name, (n_desc >> 8) & 0xFF))
    return {"cputype": cputype & 0xFFFFFFFF, "filetype": filetype, "flags": flags, "dylibs": dylibs,
            "builds": builds, "version_mins": version_mins, "undefined": undefined}


def _macho_version(v: int) -> tuple:
    return v >> 16, (v >> 8) & 0xFF, v & 0xFF


def macho_platforms(macho: dict) -> list:
    """[(platform, minos, load command)] from LC_BUILD_VERSION and the older LC_VERSION_MIN_* commands.

    LC_VERSION_MIN_IPHONEOS names no platform: an arm64 binary is a device one, an x86_64 binary a
    simulator one (arm64 simulators did not exist before LC_BUILD_VERSION).
    """
    out = [(plat, version, "LC_BUILD_VERSION") for plat, version in macho["builds"]]
    for cmd, version in macho["version_mins"]:
        if cmd == 0x25:
            plat = 7 if macho["cputype"] == CPU_TYPE_X86_64 else 2
        else:
            plat = LC_VERSION_MIN_PLATFORM[cmd]
        out.append((plat, version, f"LC_VERSION_MIN_{LC_VERSION_MIN[cmd]}"))
    return out


def describe_macho(macho: dict) -> str:
    plats = ", ".join(f"{MACHO_PLATFORMS.get(p, p)} {'.'.join(map(str, v))} ({cmd})" for p, v, cmd in macho_platforms(macho))
    imports = macho.get("undefined") or []
    python = sum(1 for _s, o in imports if 1 <= o <= len(macho["dylibs"]) and macho["dylibs"][o - 1] == PYTHON_FRAMEWORK)
    return f"{plats or 'no platform'} {macho['dylibs']}; {len(imports)} imports ({python} from Python.framework)"


def openssl_versions(data: bytes) -> set:
    """The x.y.z of every 'OpenSSL x.y.z' string in a binary.

    A statically linked libcrypto carries its own OPENSSL_VERSION_TEXT; cryptography's cffi module
    also compiles in the text of the headers it was built against. So a module built against the
    3.5.9 headers but linked with a 3.0.x libcrypto carries both, and presence alone proves nothing.
    """
    return {m.decode() for m in OPENSSL_VERSION_TEXT.findall(data)}


def _openssl_version_errors(name: str, data: bytes, openssl: str) -> list:
    versions = openssl_versions(data)
    errors = []
    if openssl not in versions:
        errors.append(f"{name}: no 'OpenSSL {openssl}' version string (static OpenSSL {openssl} not linked)")
    others = sorted(versions - {openssl})
    if others:
        errors.append(f"{name}: OpenSSL version strings {others} besides {openssl}: another libcrypto/libssl "
                      f"was linked in (the {openssl} text alone can come from the headers)")
    return errors


def _brief(names, limit: int = 12) -> str:
    names = sorted(names)
    return str(names[:limit])[:-1] + (f", ... +{len(names) - limit}]" if len(names) > limit else "]")


def check_android_binary(name: str, data: bytes, target: str, package: str, openssl: str, stubs=None) -> list:
    """Static checks of one Android module.

    Every module: ELF machine, 16 KB LOAD alignment, no forbidden NEEDED, and no undefined OpenSSL
    symbol (Android links modules BIND_NOW; dlopen fails on any import nothing provides). The
    cryptography ``_rust`` module also: NEEDED only libpython + bionic, every strong import either
    the Python C API (libpython in NEEDED) or exported by the ``stubs`` of its NEEDED system
    libraries at the wheel's API level, and exactly one OpenSSL version text (the pinned one).
    """
    errors = []
    try:
        elf = parse_elf(data)
    except (ValueError, struct.error) as e:
        return [f"{name}: {e}"]
    if elf["machine"] != ELF_MACHINE[target]:
        errors.append(f"{name}: ELF machine {elf['machine']} is not {target} ({ELF_MACHINE[target]})")
    small = [a for a in elf["load_aligns"] if a < ANDROID_PAGE]
    if not elf["load_aligns"] or small:
        errors.append(f"{name}: LOAD alignment {[hex(a) for a in elf['load_aligns']]} (need >= {ANDROID_PAGE:#x})")
    bad = [n for n in elf["needed"] if ANDROID_FORBIDDEN_NEEDED.match(n)]
    if bad:
        errors.append(f"{name}: NEEDED {bad} (libpython3.so, libcrypto/libssl and libc++_shared.so are not allowed)")
    try:
        symbols = elf_dynsym(data)
    except (ValueError, struct.error) as e:
        return errors + [f"{name}: cannot read the dynamic symbols ({e})"]
    undefined = [(n, bind) for n, bind, defined in symbols if not defined]
    ossl = [n for n, _b in undefined if OPENSSL_SYMBOL.match(n)]
    if ossl:
        errors.append(f"{name}: {len(ossl)} undefined OpenSSL symbols {_brief(ossl)}: OpenSSL is not (fully) "
                      f"linked in, and dlopen fails on them on the device")
    base = name.rsplit("/", 1)[-1]
    if package == "cryptography" and base.startswith("_rust."):
        extra = sorted(set(elf["needed"]) - CRYPTOGRAPHY_ANDROID_NEEDED)
        if extra:
            errors.append(f"{name}: NEEDED {extra} beyond {sorted(CRYPTOGRAPHY_ANDROID_NEEDED)}")
        strong = [n for n, bind in undefined if bind != STB_WEAK and not OPENSSL_SYMBOL.match(n)]
        python = [n for n in strong if PYTHON_SYMBOL.match(n)]
        if python and ANDROID_LIBPYTHON not in elf["needed"]:
            errors.append(f"{name}: imports {len(python)} Python C API symbols but does not need {ANDROID_LIBPYTHON}")
        rest = [n for n in strong if not PYTHON_SYMBOL.match(n)]
        system = [lib for lib in elf["needed"] if lib in CRYPTOGRAPHY_ANDROID_NEEDED and lib != ANDROID_LIBPYTHON]
        if rest:
            api = cmw.TARGETS[target].max_version[0]
            exports, problem = stubs.exports(target, system) if stubs is not None else (None, "no NDK given")
            if exports is None:
                errors.append(f"{name}: cannot resolve {len(rest)} undefined symbols against the NDK API {api} "
                              f"stubs of {system}: {problem}")
            else:
                missing = [n for n in rest if n not in exports]
                if missing:
                    errors.append(f"{name}: {len(missing)} undefined symbols {_brief(missing)} are exported by none "
                                  f"of its NEEDED {system} at API {api}: dlopen fails on them")
        errors += _openssl_version_errors(name, data, openssl)
    if package == "pillow" and base.startswith("_imaging.") and not any(n.startswith("libjpeg.so") for n in elf["needed"]):
        errors.append(f"{name}: does not need libjpeg.so (NEEDED {elf['needed']})")
    return errors


def describe_elf(data: bytes) -> str:
    elf, symbols = parse_elf(data), elf_dynsym(data)
    undefined = [(n, bind) for n, bind, defined in symbols if not defined]
    python = sum(1 for n, _b in undefined if PYTHON_SYMBOL.match(n))
    weak = sum(1 for _n, bind in undefined if bind == STB_WEAK)
    return f"NEEDED {elf['needed']}; {len(undefined)} imports ({python} Python C API, {weak} weak)"


def ios_minos_limit(target: str, minos: tuple) -> tuple:
    """The highest minos a slice may carry: the deployment target, or 14.0 for arm64 simulators."""
    return max(minos, IOS_ARM64_SIMULATOR_FLOOR) if target == "iphonesimulator.arm64" else minos


def check_ios_binary(name: str, data: bytes, target: str, package: str, openssl: str, minos: tuple) -> list:
    """Static checks of one iOS slice.

    The platform comes from LC_BUILD_VERSION or, for deployment targets below iOS 12, from
    LC_VERSION_MIN_IPHONEOS (what forge's Rust builds carry without IPHONEOS_DEPLOYMENT_TARGET:
    rustc's default 10.0). The arm64 simulator slice may carry 14.0 (LLVM's floor) under the
    ios_13_0 tag; the other two must not need a newer iOS than ``minos``.

    Imports: the image must use the two-level namespace, and every undefined symbol must be bound
    to one of its dylibs by library ordinal. ld64 checks those binds at link time; what forge's
    ``-undefined dynamic_lookup`` lets through unresolved gets DYNAMIC_LOOKUP_ORDINAL instead and
    is only looked for at run time. Python.framework may only provide Python C API symbols, and
    the cryptography ``_rust`` module carries exactly one OpenSSL version text (the pinned one).
    """
    errors = []
    try:
        macho = parse_macho(data)
    except (ValueError, struct.error) as e:
        return [f"{name}: {e}"]
    if not macho["flags"] & MH_TWOLEVEL:
        errors.append(f"{name}: flat namespace (MH_TWOLEVEL not set): imports are looked up by name at run time")
    if macho["undefined"] is None:
        errors.append(f"{name}: no LC_SYMTAB, so its imports cannot be checked")
    unbound, not_python, ossl = [], [], []
    dylibs = macho["dylibs"]
    for sym, ordinal in macho["undefined"] or []:
        plain = sym[1:] if sym.startswith("_") else sym          # Mach-O prefixes C names with "_"
        if OPENSSL_SYMBOL.match(plain):
            ossl.append(sym)
        if not 1 <= ordinal <= len(dylibs):
            unbound.append(f"{sym} ({MACHO_SPECIAL_ORDINALS.get(ordinal, f'ordinal {ordinal}')})")
        elif dylibs[ordinal - 1] == PYTHON_FRAMEWORK and not PYTHON_SYMBOL.match(plain):
            not_python.append(sym)
    if unbound:
        errors.append(f"{name}: {len(unbound)} undefined symbols bound to none of its dylibs {_brief(unbound)} "
                      f"(left unresolved at link time; dyld fails on them)")
    if not_python:
        errors.append(f"{name}: {len(not_python)} non-Python symbols {_brief(not_python)} bound to {PYTHON_FRAMEWORK}")
    if ossl:
        errors.append(f"{name}: {len(ossl)} undefined OpenSSL symbols {_brief(ossl)}: OpenSSL is not (fully) "
                      f"linked in")
    cputype, platform = MACHO_EXPECT[target]
    if macho["cputype"] != cputype:
        errors.append(f"{name}: cputype {macho['cputype']:#x} is not {target}")
    plats = macho_platforms(macho)
    if len(plats) != 1:
        errors.append(f"{name}: {len(plats)} platform load commands {[cmd for _p, _v, cmd in plats]} "
                      f"(expected one LC_BUILD_VERSION or LC_VERSION_MIN_IPHONEOS)")
    limit = ios_minos_limit(target, minos)
    for plat, version, cmd in plats:
        if plat != platform:
            errors.append(f"{name}: platform {MACHO_PLATFORMS.get(plat, plat)} ({cmd}) is not "
                          f"{MACHO_PLATFORMS[platform]}")
        if version[:2] > limit:
            errors.append(f"{name}: minos {'.'.join(map(str, version))} ({cmd}) is above "
                          f"{'.'.join(map(str, limit))}")
    bad = [d for d in macho["dylibs"] if not IOS_ALLOWED_DYLIB.match(d) or "clang_rt" in d]
    if bad:
        errors.append(f"{name}: links {bad} (only @rpath/Python.framework/Python and system libraries)")
    if package == "cryptography" and name.rsplit("/", 1)[-1].startswith("_rust."):
        errors += _openssl_version_errors(name, data, openssl)
        if b"clang_rt.osx" in data:
            errors.append(f"{name}: references clang_rt.osx")
    return errors


def verify_native_wheel(path: Path, package: str, target: str, openssl: str, minos: tuple, stubs=None) -> tuple:
    """(errors, notes) for one wheel. ``stubs``: NdkStubs for the Android import checks."""
    errors, notes = [], []
    with zipfile.ZipFile(path) as zf:
        sos = sorted(n for n in zf.namelist() if n.endswith(".so"))
        if not sos:
            return [f"{path.name}: no native modules"], notes
        for name in sos:
            data = zf.read(name)
            if cmw.platform_of(target) == "android":
                errs = check_android_binary(name, data, target, package, openssl, stubs)
                if not errs:
                    notes.append(f"{path.name}: {name} {describe_elf(data)}")
            else:
                errs = check_ios_binary(name, data, target, package, openssl, minos)
                if not errs:
                    notes.append(f"{path.name}: {name} {describe_macho(parse_macho(data))}")
            errors += [f"{path.name}: {e}" for e in errs]
        if package == "pillow" and cmw.platform_of(target) == "android":
            for module in PILLOW_MODULES:
                if not any(n.rsplit("/", 1)[-1].startswith(module + ".") and n.startswith("PIL/") for n in sos):
                    errors.append(f"{path.name}: no PIL/{module}*.so")
    return errors, notes


def cmd_verify_native(ctx: Ctx, args) -> int:
    wheelhouse = Path(args.wheelhouse)
    openssl, _spec = openssl_spec(ctx)
    minos = ctx.ios_deployment_target
    stubs = None
    if ctx.platform == "android":
        ndk = Path(args.ndk) if args.ndk else default_ndk(ctx)
        stubs = NdkStubs(ndk)
        print(f"NDK for the import checks: {ndk or 'none (pass --ndk, or set NDK_HOME or ANDROID_HOME)'}")
    errors, notes = [], []
    for w, target in ctx.manifest.promised(ctx.platform):
        matches = find_wheel(wheelhouse, w.filename(target, ctx.manifest.build_tag))
        if len(matches) != 1:
            errors.append(f"{wheelhouse}: expected one {w.filename(target, ctx.manifest.build_tag)}")
            continue
        errs, ns = verify_native_wheel(matches[0], w.name, target, openssl, minos, stubs)
        errors += errs
        notes += ns
    for n in notes:
        print(f"ok      {n}")
    for e in errors:
        print(f"ERROR   {e}")
    print(f"verify-native ({ctx.platform}): {len(errors)} errors")
    return 1 if errors else 0


# ============================================================================ device self-test report
def check_selftest_report(report: dict, platform: str, crypto_version: str, openssl: str,
                          pillow_version: str) -> list:
    """The smoke suite must show the self-built cryptography (with OpenSSL ``openssl``) and Pillow."""
    errors = []
    if report.get("suite") != "smoke":
        errors.append(f"suite is {report.get('suite')!r}, expected 'smoke' (the smoke suite's report or "
                      f"{WHEELS_MARKER} line)")
    if report.get("platform") != platform:
        errors.append(f"platform is {report.get('platform')!r}, expected {platform!r}")
    if not report.get("strict"):
        errors.append("the report is not from a strict (device) run")
    checks = {c.get("name"): c for c in report.get("checks") or [] if isinstance(c, dict)}

    def passed(name: str) -> dict | None:
        c = checks.get(name)
        if c is None:
            errors.append(f"check {name!r} is missing from the report")
            return None
        if c.get("status") != "pass":
            errors.append(f"check {name!r} is {c.get('status')}: {c.get('error') or c.get('reason')}")
            return None
        return c.get("detail") or {}

    fernet = passed("fernet")
    if fernet is not None:
        if fernet.get("version") != crypto_version:
            errors.append(f"cryptography {fernet.get('version')!r} on the device, expected {crypto_version}")
        text = fernet.get("openssl") or ""
        if not re.match(r"^OpenSSL " + re.escape(openssl) + r"(\s|$)", text):
            errors.append(f"cryptography's OpenSSL is {text!r}, expected OpenSSL {openssl}")
    pillow = passed("pillow")
    if pillow is not None:
        if pillow.get("version") != pillow_version:
            errors.append(f"Pillow {pillow.get('version')!r} on the device, expected {pillow_version}")
        features = pillow.get("features") or {}
        missing = [f for f in PILLOW_FEATURES if features.get(f) is not True]
        if missing:
            errors.append(f"Pillow features missing: {missing} (reported {features})")
        roundtrip = pillow.get("roundtrip") or {}
        bad = [f for f in PILLOW_ROUNDTRIP if not roundtrip.get(f)]
        if bad:
            errors.append(f"Pillow round-trip failed for {bad} (reported {roundtrip})")
        if pillow.get("font") != "FreeTypeFont":
            errors.append(f"ImageFont.load_default(size=24) gave {pillow.get('font')!r}, expected FreeTypeFont")
    return errors


def _wheels_payloads(text: str) -> list:
    """The JSON payloads of the GLOSSARION_WHEELS lines in a device log, oldest first."""
    found = []
    for line in text.splitlines():
        m = WHEELS_LINE.search(line)
        if m:
            found.append(m.group(1).rstrip())
    return found


def load_selftest_evidence(paths) -> tuple:
    """(report or None, where it came from or the reasons each input gave nothing).

    Each input is either a self-test report (``selftest-smoke.json``, the iOS simulator's app
    container) or a device log holding the smoke suite's ``GLOSSARION_WHEELS {json}`` marker line
    (logcat on Android: ``flet build apk`` makes release-mode APKs, debug-signed or not, so
    ``run-as`` cannot read the app's files). The first input that yields a report wins; in a log,
    the last marker of the smoke suite does.
    """
    tried = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            tried.append(f"{p}: not found")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        body = text.lstrip("﻿ \t\r\n")
        if body.startswith("{"):
            try:
                data = json.loads(body)
            except ValueError as e:
                tried.append(f"{p}: not a valid JSON report ({e})")
                continue
            if isinstance(data, dict):
                return data, f"{p} (self-test report)"
            tried.append(f"{p}: not a JSON object")
            continue
        payloads = _wheels_payloads(text)
        if not payloads:
            tried.append(f"{p}: no {WHEELS_MARKER} line")
            continue
        parsed = []
        for raw in payloads:
            try:
                data = json.loads(raw)
            except ValueError:
                continue
            if isinstance(data, dict):
                parsed.append(data)
        if not parsed:
            tried.append(f"{p}: {len(payloads)} {WHEELS_MARKER} line(s), none valid JSON (cut off by the log?)")
            continue
        smoke = [d for d in parsed if d.get("suite") == "smoke"]
        return (smoke or parsed)[-1], f"{p} ({WHEELS_MARKER} line, {len(payloads)} found)"
    return None, tried


def cmd_assert_selftest(ctx: Ctx, args) -> int:
    report, source = load_selftest_evidence(args.report)
    if report is None:
        for reason in source:
            print(f"::error title=Mobile wheels on device::{reason}" if os.environ.get("GITHUB_ACTIONS")
                  else f"ERROR   {reason}")
        print(f"assert-selftest ({ctx.platform}): no self-test report or {WHEELS_MARKER} line: FAIL")
        return 1
    crypto = ctx.manifest.wheel("cryptography")
    if crypto is None:
        raise WheelsError(f"{ctx.manifest_path.name} does not build cryptography")
    pins = {name: pinned_version(ctx.pyproject, name) for name in ("cryptography", "pillow")}
    if pins["cryptography"] != crypto.version:
        raise WheelsError(f"pyproject pins cryptography {pins['cryptography']}, the manifest builds {crypto.version}")
    if pins["pillow"] is None:
        raise WheelsError("pyproject does not pin pillow with ==")
    pillow = ctx.manifest.wheel("pillow")
    if pillow is not None and pillow.version != pins["pillow"]:
        raise WheelsError(f"pyproject pins pillow {pins['pillow']}, the manifest builds {pillow.version}")
    openssl, _spec = openssl_spec(ctx)
    errors = check_selftest_report(report, ctx.platform, crypto.version, openssl, pins["pillow"])
    for e in errors:
        print(f"::error title=Mobile wheels on device::{e}" if os.environ.get("GITHUB_ACTIONS") else f"ERROR   {e}")
    print(f"assert-selftest ({ctx.platform}, {source}): cryptography {crypto.version} with OpenSSL {openssl}, "
          f"pillow {pins['pillow']}: {'FAIL' if errors else 'PASS'}")
    return 1 if errors else 0


# ============================================================================ CLI
def cmd_env(ctx: Ctx, args) -> None:
    tc = ctx.toolchain
    targets = ctx.targets()
    values = {
        "PACKAGES": " ".join(w.name for w in ctx.wheels()),
        "FORGE_REPO": tc["mobile_forge"]["repo"],
        "FORGE_REV": tc["mobile_forge"]["rev"],
        "CROSSENV_REPO": tc["crossenv"]["repo"],
        "CROSSENV_REV": tc["crossenv"]["rev"],
        "PYTHON_BUILD_RELEASE": str(tc["python_build_release"]),
        "PY_FULL": ctx.py_full,
        "PY_SHORT": ctx.py_short,
        "FORGE_HOST": FORGE_HOST[ctx.platform],
        "RUST_VERSION": str(tc["rust"]),
        "RUST_TARGETS": ",".join(RUST_TARGETS[t] for t in targets),
        "WHEEL_BUILD_TAG": str(ctx.manifest.build_tag),
    }
    if ctx.platform == "android":
        values["ANDROID_ABIS"] = " ".join(targets)
        values["ANDROID_NDK_VERSION"] = str(tc["android_ndk"])
    # No IPHONEOS_DEPLOYMENT_TARGET / OPENSSL_STATIC here: forge's compile_env() never passes the
    # outer environment to the build, so those live in the recipes' script_env (checked below).
    for w in ctx.wheels():
        ctx.recipe_dir(w)            # recipe version / build number / iOS deployment target match the manifest
    if args.constraints_out:
        constraints = tc.get("pip_constraints") or {}
        path = Path(args.constraints_out).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{k}=={v}\n" for k, v in constraints.items()), encoding="utf-8")
        values["PIP_CONSTRAINT"] = str(path)
    if args.format == "github":
        for k, v in values.items():
            print(f"{k.lower()}={v}")
    else:
        for k, v in values.items():
            print(f"export {k}={shlex.quote(v)}")


def cmd_fetch(ctx: Ctx, args) -> None:
    for url, path, sha in fetch_plan(ctx, args.packages, Path(args.dest)):
        download_verified(url, path, sha)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Self-built mobile wheels: build helpers and checks (see docstring).")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="wheelhouse.toml")
    ap.add_argument("--pyproject", default=str(DEFAULT_PYPROJECT), help="the mobile pyproject.toml (pins, uv.lock)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, help_text, packages=False):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--platform", required=True, choices=PLATFORMS)
        if packages:
            p.add_argument("packages", nargs="*", help="packages (default: all the manifest builds for the platform)")
        return p

    add("packages", "print the packages built for the platform")
    p = add("recipe", "print a package's recipe directory")
    p.add_argument("package")
    p = add("env", "print the pinned build inputs")
    p.add_argument("--format", choices=("shell", "github"), default="shell")
    p.add_argument("--constraints-out", help="write the pip constraints file here and export PIP_CONSTRAINT")
    p = add("fetch", "download and sha256-check the build inputs", packages=True)
    p.add_argument("--dest", required=True, help="forge's downloads/ directory")
    p = add("stage-openssl", "swap in the static BeeWare OpenSSL", packages=True)
    p.add_argument("--downloads", required=True)
    p.add_argument("--forge", required=True, help="the mobile-forge checkout (cwd of setup.sh)")
    p = add("collect", "copy the promised wheels from forge's dist/ into the wheel cache", packages=True)
    p.add_argument("--dist", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--logs", help="forge's logs/ directory (maturin version for build-info.json)")
    p = add("assemble", "copy the cached wheels into the wheelhouse and write SHA256SUMS")
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p = add("provenance", "write provenance.json into the wheelhouse")
    p.add_argument("--wheelhouse", required=True)
    p.add_argument("--cache")
    p.add_argument("--built", default="", help="packages built in this run (the rest came from the cache)")
    p = add("verify-native", "static checks of the native binaries")
    p.add_argument("--ndk", help="Android NDK whose API-level stub libraries resolve the imports "
                                 "(default: $NDK_HOME, else $ANDROID_HOME/ndk/<toolchain.android_ndk>)")
    p.add_argument("wheelhouse")
    p = add("assert-selftest", "check the device self-test (a report, or a log with the GLOSSARION_WHEELS line)")
    p.add_argument("report", nargs="+",
                   help="selftest-smoke.json or a device log (logcat); the first that yields a report is used")
    args = ap.parse_args(argv)

    try:
        ctx = Ctx(args.manifest, args.platform, args.pyproject)
        if args.cmd == "packages":
            print(" ".join(w.name for w in ctx.wheels()))
        elif args.cmd == "recipe":
            print(ctx.recipe_dir(ctx.wheels([args.package])[0]))
        elif args.cmd == "env":
            cmd_env(ctx, args)
        elif args.cmd == "fetch":
            cmd_fetch(ctx, args)
        elif args.cmd == "stage-openssl":
            cmd_stage_openssl(ctx, args)
        elif args.cmd == "collect":
            cmd_collect(ctx, args)
        elif args.cmd == "assemble":
            cmd_assemble(ctx, args)
        elif args.cmd == "provenance":
            cmd_provenance(ctx, args)
        elif args.cmd == "verify-native":
            return cmd_verify_native(ctx, args)
        elif args.cmd == "assert-selftest":
            return cmd_assert_selftest(ctx, args)
    except (WheelsError, cmw.ManifestError, OSError, KeyError, ValueError, subprocess.CalledProcessError) as e:
        print(f"wheels.py {args.cmd}: error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
