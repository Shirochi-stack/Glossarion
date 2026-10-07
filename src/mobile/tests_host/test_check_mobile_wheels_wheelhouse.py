"""Offline tests for the self-built mobile wheels.

* tools/check_mobile_wheels.py: --wheelhouse / --wheelhouse-plan / --verify-wheelhouse /
  --verify-installed / --platform, pip's ranking, the manifest checks, and that without the
  new flags nothing changes (the pins keep failing while no index has their mobile wheels).
* ci/wheels/wheels.py: ELF / Mach-O checks (including the load commands forge's Rust builds
  really carry), the self-test assertion (JSON report or the logcat GLOSSARION_WHEELS line),
  OpenSSL staging helpers, collect / assemble, the pinned inputs.
* ci/wheels/build_wheels.sh: run end to end with git, uv, setup.sh and forge stubbed (needs bash),
  asserting every wheels.py and forge call's exact arguments.
* The committed ci/wheels files agree with pyproject.toml, uv.lock and mobile-forge upstream.

Everything runs on synthetic wheels in tmp_path and a fake index (Fetcher._get is replaced),
so no network is used.

Run: python -m pytest -p no:cacheprovider -W ignore tests_host/test_check_mobile_wheels_wheelhouse.py
"""
from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.util
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

MOBILE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOBILE / "tools"))
cmw = importlib.import_module("check_mobile_wheels")
_spec = importlib.util.spec_from_file_location("mobile_ci_wheels", MOBILE / "ci" / "wheels" / "wheels.py")
wh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wh)

CRYPTO_SHA = "7b46165bb56eb4704e2eaaf86f3c940d19154535d9b0ca7d6d590b04060e00d5"
PILLOW_SHA = "3b8182a766685eaa002637e28b4ec8d6b18819a0c71f579bf0dbaa5830297cce"
CRYPTO_REQS = ['cffi>=2.0.0; platform_python_implementation != "PyPy"',
               'typing-extensions>=4.13.2; python_full_version < "3.11"']
PILLOW_REQS = ["flet-libjpeg==3.0.90", "flet-libfreetype==2.13.3", "flet-libwebp==1.6.0"]
ANDROID = ["arm64-v8a", "x86_64"]
IOS = ["iphoneos.arm64", "iphonesimulator.arm64", "iphonesimulator.x86_64"]
LABEL = {k: t.label for k, t in cmw.TARGETS.items()}

MANIFEST = f"""
schema = 1
build_tag = 1000

[toolchain]
python_build_release = "20260921"
python_version = "3.13.15"
android_ndk = "27.3.13750724"
rust = "1.98.1"
ios_deployment_target = "13.0"
mobile_forge = {{ repo = "https://github.com/flet-dev/mobile-forge", rev = "{'a' * 40}" }}
crossenv = {{ repo = "https://github.com/flet-dev/crossenv", rev = "{'b' * 40}" }}
support_sha256 = {{ android = "{'1' * 64}", ios = "{'2' * 64}" }}
pip_constraints = {{ cffi = "2.0.0", maturin = "1.15.0" }}

[openssl]
version = "3.5.9"
[openssl.android]
repo = "beeware/cpython-android-source-deps"
tag = "openssl-3.5.9-0"
build = "0"
[openssl.android.assets]
"arm64-v8a" = {{ file = "openssl-3.5.9-0-aarch64-linux-android.tar.gz", sha256 = "{'3' * 64}" }}
"x86_64" = {{ file = "openssl-3.5.9-0-x86_64-linux-android.tar.gz", sha256 = "{'4' * 64}" }}
[openssl.ios]
repo = "beeware/cpython-apple-source-deps"
tag = "OpenSSL-3.5.9-1"
build = "1"
[openssl.ios.assets]
"iphoneos.arm64" = {{ file = "openssl-3.5.9-1-iphoneos.arm64.tar.gz", sha256 = "{'5' * 64}" }}
"iphonesimulator.arm64" = {{ file = "openssl-3.5.9-1-iphonesimulator.arm64.tar.gz", sha256 = "{'6' * 64}" }}
"iphonesimulator.x86_64" = {{ file = "openssl-3.5.9-1-iphonesimulator.x86_64.tar.gz", sha256 = "{'7' * 64}" }}

[[wheel]]
name = "cryptography"
version = "50.0.2"
recipe = "recipes/cryptography"
sdist_sha256 = "{CRYPTO_SHA}"
targets = {json.dumps(ANDROID + IOS)}
requires_dist = {json.dumps(CRYPTO_REQS)}

[[wheel]]
name = "pillow"
version = "12.3.0"
recipe = "recipes/pillow"
sdist_sha256 = "{PILLOW_SHA}"
targets = {json.dumps(ANDROID)}
requires_dist = {json.dumps(PILLOW_REQS)}
"""

PYPROJECT = """
[project]
name = "fixture"
version = "1.0"
dependencies = [
  "cryptography==50.0.2",
  "pillow==12.3.0",
  "app-pure==1.0",
]

[tool.flet.android]
target_arch = ["arm64-v8a", "x86_64"]
"""

UV_LOCK = f"""
version = 1

[[package]]
name = "cryptography"
version = "50.0.2"
sdist = {{ url = "https://files.pythonhosted.org/x/cryptography-50.0.2.tar.gz", hash = "sha256:{CRYPTO_SHA}" }}

[[package]]
name = "pillow"
version = "12.3.0"
sdist = {{ url = "https://files.pythonhosted.org/x/pillow-12.3.0.tar.gz", hash = "sha256:{PILLOW_SHA}" }}
"""


# --------------------------------------------------------------------------- fixtures / builders
def _record_line(path: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{path},sha256={digest},{len(data)}"


def make_wheel(directory: Path, name: str, version: str, tag: str, build: str = "1000", requires=(),
               files: dict | None = None, wheel_build: str | None = "same", wheel_tags=None,
               meta_name: str | None = None) -> Path:
    """A minimal valid wheel; the keyword arguments break it in specific ways."""
    dist = name.replace("-", "_")
    filename = f"{dist}-{version}-{build}-{tag}.whl" if build else f"{dist}-{version}-{tag}.whl"
    dist_info = f"{dist}-{version}.dist-info"
    members = dict(files if files is not None else {f"{dist}/__init__.py": b"x = 1\n"})
    meta = f"Metadata-Version: 2.1\nName: {meta_name or name}\nVersion: {version}\n"
    meta += "".join(f"Requires-Dist: {r}\n" for r in requires) + "\nlong description\n"
    members[f"{dist_info}/METADATA"] = meta.encode()
    build_line = build if wheel_build == "same" else wheel_build
    wheel = "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\n"
    wheel += f"Build: {build_line}\n" if build_line else ""
    wheel += "".join(f"Tag: {t}\n" for t in (wheel_tags or [tag]))
    members[f"{dist_info}/WHEEL"] = wheel.encode()
    record = [_record_line(p, d) for p, d in members.items()] + [f"{dist_info}/RECORD,,"]
    members[f"{dist_info}/RECORD"] = ("\n".join(record) + "\n").encode()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p, d in members.items():
            zf.writestr(p, d)
    return path


def full_wheelhouse(directory: Path, targets_crypto=None, targets_pillow=None) -> Path:
    for t in targets_crypto if targets_crypto is not None else ANDROID + IOS:
        make_wheel(directory, "cryptography", "50.0.2", f"cp313-cp313-{LABEL[t]}",
                   requires=CRYPTO_REQS + ['bcrypt>=3.1.5; extra == "ssh"'])
    for t in targets_pillow if targets_pillow is not None else ANDROID:
        make_wheel(directory, "pillow", "12.3.0", f"cp313-cp313-{LABEL[t]}",
                   requires=["flet-libjpeg (==3.0.90)", "flet-libfreetype (==2.13.3)", "flet-libwebp (==1.6.0)",
                             'furo; extra == "docs"'])
    return directory


class FakeIndex:
    """PyPI simple/JSON + pypi.flet.dev served from memory (stands in for Fetcher._get)."""

    def __init__(self):
        self.pypi: dict[str, list] = {}
        self.flet: dict[str, list] = {}
        self.meta: dict[tuple, list] = {}

    def add(self, source: str, name: str, *filenames: str, requires=None, version=None):
        getattr(self, source).setdefault(name, []).extend(filenames)
        if requires is not None:
            self.meta[(name, version)] = list(requires)

    def get(self, fetcher, url: str, accept=None):
        fetcher.requests += 1
        if url == f"{cmw.FLET_INDEX}/":
            links = "".join(f'<a href="{cmw.FLET_INDEX}/{n}/">{n}</a>\n' for n in self.flet)
            return 200, f"<html><body>{links}</body></html>".encode()
        m = re.fullmatch(re.escape(cmw.PYPI_SIMPLE) + r"/([^/]+)/", url)
        if m:
            files = self.pypi.get(m.group(1))
            if files is None:
                return 404, b""
            return 200, json.dumps({"files": [{"filename": f, "requires-python": ">=3.9", "yanked": False}
                                              for f in files]}).encode()
        m = re.fullmatch(re.escape(cmw.FLET_INDEX) + r"/([^/]+)/", url)
        if m:
            files = self.flet.get(m.group(1), [])
            return 200, "".join(f'<a href="{cmw.FLET_INDEX}/files/{f}#sha256=0">{f}</a>\n' for f in files).encode()
        m = re.fullmatch(re.escape(cmw.PYPI_JSON) + r"/([^/]+)/([^/]+)/json", url)
        if m:
            reqs = self.meta.get((m.group(1), m.group(2)))
            if reqs is None:
                return 404, b""
            return 200, json.dumps({"info": {"requires_dist": reqs}}).encode()
        return 404, b""


def default_index() -> FakeIndex:
    ix = FakeIndex()
    ix.add("pypi", "cryptography", "cryptography-50.0.2.tar.gz", "cryptography-50.0.2-cp311-abi3-manylinux_2_28_x86_64.whl",
           requires=CRYPTO_REQS + ['bcrypt>=3.1.5; extra == "ssh"'], version="50.0.2")
    ix.add("flet", "cryptography", *(f"cryptography-43.0.1-10-cp313-cp313-{LABEL[t]}.whl" for t in ANDROID + IOS))
    ix.add("pypi", "cffi", "cffi-2.1.1-cp313-cp313-manylinux_2_17_x86_64.whl", requires=["pycparser"], version="2.1.1")
    ix.add("flet", "cffi", *(f"cffi-2.0.0-1-cp313-cp313-{LABEL[t]}.whl" for t in ANDROID + IOS))
    ix.meta[("cffi", "2.0.0")] = ["pycparser; implementation_name != 'PyPy'"]
    ix.add("pypi", "pycparser", "pycparser-2.23-py3-none-any.whl", requires=[], version="2.23")
    ix.add("pypi", "pillow", "pillow-12.3.0.tar.gz", "pillow-12.3.0-cp313-cp313-manylinux_2_28_x86_64.whl",
           *(f"pillow-12.3.0-cp313-cp313-{LABEL[t]}.whl" for t in IOS),
           requires=['furo; extra == "docs"'], version="12.3.0")
    ix.add("flet", "pillow", *(f"pillow-12.2.0-2-cp313-cp313-{LABEL[t]}.whl" for t in ANDROID))
    for lib, ver, build in (("flet-libjpeg", "3.0.90", "1"), ("flet-libfreetype", "2.13.3", "10"),
                            ("flet-libwebp", "1.6.0", "1")):
        ix.add("flet", lib, *(f"{lib.replace('-', '_')}-{ver}-{build}-py3-none-{LABEL[t]}.whl" for t in ANDROID))
        ix.meta[(lib, ver)] = []
    ix.add("pypi", "app-pure", "app_pure-1.0-py3-none-any.whl", requires=[], version="1.0")
    return ix


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """pyproject + uv.lock + wheelhouse.toml in tmp_path, and the fake index behind Fetcher._get."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (root / "uv.lock").write_text(UV_LOCK, encoding="utf-8")
    (root / "wheelhouse.toml").write_text(MANIFEST, encoding="utf-8")
    index = default_index()
    monkeypatch.setattr(cmw.Fetcher, "_get", lambda self, url, accept=None: index.get(self, url, accept))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    return {"root": root, "index": index, "tmp": tmp_path}


def run_main(proj, *extra, platform=None):
    root = proj["root"]
    out = proj["tmp"] / f"result-{len(list(proj['tmp'].glob('result-*')))}.json"
    argv = ["--pyproject", str(root / "pyproject.toml"), "--wheelhouse-manifest", str(root / "wheelhouse.toml"),
            "--no-host", "--json", str(out), "-q"]
    if platform:
        argv += ["--platform", platform]
    rc = cmw.main(argv + list(extra))
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    return rc, data


def errors_text(data) -> str:
    return "\n".join(data.get("errors", []))


# --------------------------------------------------------------------------- unchanged default behaviour
def test_without_flags_the_pins_fail_closed(proj):
    rc, data = run_main(proj)
    assert rc == 1
    errs = errors_text(data)
    for target in ("android-arm64", "android-x86_64", "ios-device", "ios-sim-arm64", "ios-sim-x86_64"):
        assert f"[{target}] cryptography" in errs
    assert "[android-arm64] pillow" in errs and "[ios-device] pillow" not in errs
    assert "wheelhouse" not in data  # no new keys in the default JSON
    assert not any("[wheelhouse]" in e for e in data["errors"])


def test_without_flags_a_wheelhouse_on_disk_is_ignored(proj):
    full_wheelhouse(proj["tmp"] / "wh")
    rc, data = run_main(proj)
    assert rc == 1 and "[android-arm64] cryptography" in errors_text(data)


def test_legacy_ranking_still_prefers_flet_files():
    target = cmw.TARGETS["arm64-v8a"]

    class F:
        def files(self, name):
            return [cmw.parse_filename("lib-1.0-cp313-cp313-android_24_arm64_v8a.whl", "lib", "pypi"),
                    cmw.parse_filename("lib-1.0-1-cp313-cp313-android_24_arm64_v8a.whl", "lib", "flet")]

    legacy = cmw.Resolver(target, F(), set(), set()).pick("lib")
    assert legacy.file.source == "flet"


# --------------------------------------------------------------------------- pip ranking
def test_tag_priority_follows_pip():
    t = cmw.TARGETS["arm64-v8a"]

    def prio(fn):
        return cmw.tag_priority(cmw.parse_filename(fn, "x", "pypi"), t)

    order = ["x-1-cp313-cp313-android_24_arm64_v8a.whl", "x-1-cp313-cp313-android_21_arm64_v8a.whl",
             "x-1-cp313-abi3-android_24_arm64_v8a.whl", "x-1-cp312-abi3-android_24_arm64_v8a.whl",
             "x-1-cp37-abi3-android_24_arm64_v8a.whl", "x-1-py3-none-android_24_arm64_v8a.whl",
             "x-1-py312-none-android_24_arm64_v8a.whl", "x-1-py3-none-any.whl", "x-1-py312-none-any.whl"]
    prios = [prio(fn) for fn in order]
    assert all(p is not None for p in prios)
    assert prios == sorted(prios, reverse=True)
    assert prio("x-1-cp313-cp313-android_26_arm64_v8a.whl") is None  # above API 24
    assert prio("x-1-cp313-cp313-android_24_x86_64.whl") is None


def test_pip_rank_build_tag_breaks_ties_and_index_wins_exact_ties():
    t = cmw.TARGETS["iphoneos.arm64"]
    local = cmw.parse_filename("pillow-12.3.0-1000-cp313-cp313-ios_13_0_arm64_iphoneos.whl", "pillow", "wheelhouse")
    index_same = cmw.parse_filename("pillow-12.3.0-1000-cp313-cp313-ios_13_0_arm64_iphoneos.whl", "pillow", "pypi")
    index_plain = cmw.parse_filename("pillow-12.3.0-cp313-cp313-ios_13_0_arm64_iphoneos.whl", "pillow", "pypi")
    index_more = cmw.parse_filename("pillow-12.3.0-1001-cp313-cp313-ios_13_0_arm64_iphoneos.whl", "pillow", "flet")
    assert local.build == (1000, "") and index_plain.build == ()

    def best(*files):
        return max(files, key=lambda f: cmw.pip_rank(f, t))

    assert best(local, index_plain) is local
    assert best(local, index_same) is index_same
    assert best(local, index_more) is index_more
    # a better tag beats any build tag
    abi3 = cmw.parse_filename("pillow-12.3.0-1000-cp311-abi3-ios_13_0_arm64_iphoneos.whl", "pillow", "wheelhouse")
    assert best(abi3, index_plain) is index_plain


# --------------------------------------------------------------------------- plan / wheelhouse modes
def test_plan_mode_passes_and_checks_transitive_dependencies(proj):
    rc, data = run_main(proj, "--wheelhouse-plan")
    assert rc == 0, errors_text(data)
    arm = data["targets"]["android-arm64"]
    assert arm["cryptography"]["source"] == "planned"
    assert arm["cryptography"]["file"] == "cryptography-50.0.2-1000-cp313-cp313-android_24_arm64_v8a.whl"
    assert arm["pillow"]["source"] == "planned"
    for lib in ("flet-libjpeg", "flet-libfreetype", "flet-libwebp"):   # from the planned metadata
        assert arm[lib]["source"] == "flet" and "pillow" in arm[lib]["parents"]
    assert arm["cffi"]["version"] == "2.0.0"
    assert "typing-extensions" not in arm
    assert data["targets"]["ios-device"]["pillow"]["source"] == "pypi"
    assert {row["status"] for row in data["wheelhouse"]} == {"ok"}


def test_plan_mode_fails_on_missing_transitive_wheel(proj):
    proj["index"].flet["flet-libwebp"] = []
    rc, data = run_main(proj, "--wheelhouse-plan")
    assert rc == 1 and "flet-libwebp" in errors_text(data)


def test_wheelhouse_mode_passes(proj):
    whdir = full_wheelhouse(proj["tmp"] / "wh")
    report = proj["tmp"] / "report.md"
    rc, data = run_main(proj, "--wheelhouse", str(whdir), "--report", str(report))
    assert rc == 0, errors_text(data)
    for target in ("android-arm64", "android-x86_64", "ios-device", "ios-sim-arm64", "ios-sim-x86_64"):
        assert data["targets"][target]["cryptography"]["source"] == "wheelhouse"
    assert data["targets"]["android-x86_64"]["pillow"]["file"] == "pillow-12.3.0-1000-cp313-cp313-android_24_x86_64.whl"
    text = report.read_text(encoding="utf-8")
    assert "(wheelhouse)" in text and "Self-built wheels" in text


def test_platform_filter(proj):
    whdir = full_wheelhouse(proj["tmp"] / "wh", targets_crypto=ANDROID)
    rc, data = run_main(proj, "--wheelhouse", str(whdir), platform="android")
    assert rc == 0, errors_text(data)
    assert set(data["targets"]) == {"android-arm64", "android-x86_64"}
    rc, data = run_main(proj, "--wheelhouse", str(whdir))
    assert rc == 1
    assert "missing cryptography-50.0.2-1000-cp313-cp313-ios_13_0_arm64_iphoneos.whl" in errors_text(data)


@pytest.mark.parametrize("build", ["1000", "1001"])
def test_index_file_that_ties_or_outranks_is_an_error(proj, build):
    proj["index"].add("flet", "cryptography", f"cryptography-50.0.2-{build}-cp313-cp313-android_24_arm64_v8a.whl")
    rc, data = run_main(proj, "--wheelhouse", str(full_wheelhouse(proj["tmp"] / "wh")))
    assert rc == 1
    assert (f"[android-arm64] cryptography-50.0.2-{build}-cp313-cp313-android_24_arm64_v8a.whl (flet) outranks or "
            f"ties wheelhouse") in errors_text(data)
    assert "(raise build_tag)" in errors_text(data)
    assert "[android-x86_64]" not in errors_text(data)


def test_index_file_with_lower_build_tag_loses(proj):
    proj["index"].add("flet", "cryptography", "cryptography-50.0.2-11-cp313-cp313-android_24_arm64_v8a.whl")
    rc, data = run_main(proj, "--wheelhouse", str(full_wheelhouse(proj["tmp"] / "wh")))
    assert rc == 0, errors_text(data)
    assert data["targets"]["android-arm64"]["cryptography"]["source"] == "wheelhouse"


def test_missing_target_is_an_error(proj):
    whdir = full_wheelhouse(proj["tmp"] / "wh", targets_crypto=["arm64-v8a"] + IOS)
    rc, data = run_main(proj, "--wheelhouse", str(whdir))
    assert rc == 1
    errs = errors_text(data)
    assert "missing cryptography-50.0.2-1000-cp313-cp313-android_24_x86_64.whl" in errs
    assert "[android-x86_64] cryptography" in errs


def test_pin_mismatch_is_an_error(proj):
    pp = proj["root"] / "pyproject.toml"
    pp.write_text(PYPROJECT.replace("cryptography==50.0.2", "cryptography==50.0.3"), encoding="utf-8")
    rc, data = run_main(proj, "--wheelhouse-plan")
    assert rc == 1
    errs = errors_text(data)
    assert "cryptography: wheelhouse.toml builds 50.0.2 but pyproject has 'cryptography==50.0.3'" in errs


def test_unpinned_range_is_an_error(proj):
    pp = proj["root"] / "pyproject.toml"
    pp.write_text(PYPROJECT.replace("pillow==12.3.0", "pillow>=12.3.0"), encoding="utf-8")
    rc, data = run_main(proj, "--wheelhouse-plan")
    assert rc == 1 and "pyproject has 'pillow>=12.3.0'" in errors_text(data)


def test_uv_lock_sdist_mismatch_is_an_error(proj):
    lock = proj["root"] / "uv.lock"
    lock.write_text(UV_LOCK.replace(PILLOW_SHA, "f" * 64), encoding="utf-8")
    rc, data = run_main(proj, "--wheelhouse-plan")
    assert rc == 1 and "pillow 12.3.0: sdist_sha256" in errors_text(data)
    lock.unlink()
    rc, data = run_main(proj, "--wheelhouse-plan")
    assert rc == 1 and "uv.lock: not found" in errors_text(data)


def test_wrong_requires_dist_is_an_error(proj):
    whdir = full_wheelhouse(proj["tmp"] / "wh", targets_pillow=["x86_64"])
    make_wheel(whdir, "pillow", "12.3.0", "cp313-cp313-android_24_arm64_v8a",
               requires=["flet-libjpeg==3.0.90", "flet-libfreetype==2.13.3"])
    rc, data = run_main(proj, "--wheelhouse", str(whdir))
    assert rc == 1
    assert "pillow-12.3.0-1000-cp313-cp313-android_24_arm64_v8a.whl: Requires-Dist without extras" in errors_text(data)


def test_wrong_build_tag_and_tag_are_errors(proj):
    whdir = full_wheelhouse(proj["tmp"] / "wh", targets_crypto=ANDROID[1:] + IOS)
    make_wheel(whdir, "cryptography", "50.0.2", "cp313-cp313-android_24_arm64_v8a", build="999",
               requires=CRYPTO_REQS)
    make_wheel(whdir / "x", "cryptography", "50.0.2", "cp313-cp313-android_24_x86_64",
               wheel_tags=["cp313-cp313-android_24_arm64_v8a"], requires=CRYPTO_REQS)
    rc, data = run_main(proj, "--wheelhouse", str(whdir))
    assert rc == 1
    errs = errors_text(data)
    assert "build tag 999 is not build_tag 1000" in errs
    assert "unexpected directory x/" in errs
    rc, data = run_main(proj, "--wheelhouse", str(whdir / "x"), platform="android")
    assert "WHEEL Tag ['cp313-cp313-android_24_arm64_v8a'] does not match the filename tags" in errors_text(data)


def test_unknown_files_in_the_wheelhouse_are_errors(proj):
    whdir = full_wheelhouse(proj["tmp"] / "wh")
    make_wheel(whdir, "requests", "2.99.0", "py3-none-any")
    make_wheel(whdir, "cryptography", "50.0.1", "cp313-cp313-android_24_arm64_v8a", requires=CRYPTO_REQS)
    (whdir / "evil-1.0.tar.gz").write_bytes(b"not really")
    (whdir / "SHA256SUMS").write_text("", encoding="utf-8")
    rc, data = run_main(proj, "--wheelhouse", str(whdir))
    assert rc == 1
    errs = errors_text(data)
    assert "unknown wheel requests-2.99.0-1000-py3-none-any.whl" in errs
    assert "unknown wheel cryptography-50.0.1-1000-cp313-cp313-android_24_arm64_v8a.whl" in errs
    assert "unexpected file evil-1.0.tar.gz" in errs
    assert "SHA256SUMS does not list" in errs
    # the unknown wheels never reach the resolver
    assert data["targets"]["android-arm64"].get("requests") is None


def test_flag_combinations_are_usage_errors(proj, tmp_path):
    with pytest.raises(SystemExit) as e:
        cmw.main(["--wheelhouse", str(tmp_path), "--wheelhouse-plan"])
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        cmw.main(["--verify-wheelhouse", str(tmp_path), "--wheelhouse", str(tmp_path)])
    bad = proj["root"] / "bad.toml"
    bad.write_text("schema = 2\n", encoding="utf-8")
    assert cmw.main(["--pyproject", str(proj["root"] / "pyproject.toml"), "--wheelhouse-plan",
                     "--wheelhouse-manifest", str(bad)]) == 2


# --------------------------------------------------------------------------- --verify-wheelhouse / --verify-installed
def verify(proj, directory, platform="all"):
    root = proj["root"]
    return cmw.main(["--pyproject", str(root / "pyproject.toml"), "--wheelhouse-manifest", str(root / "wheelhouse.toml"),
                     "--verify-wheelhouse", str(directory), "--platform", platform])


def write_sums(directory: Path):
    lines = [f"{cmw.file_sha256(p)}  {p.name}" for p in sorted(directory.glob("*.whl"))]
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_verify_wheelhouse_ok(proj, capsys):
    whdir = full_wheelhouse(proj["tmp"] / "wh", targets_crypto=ANDROID)
    write_sums(whdir)
    (whdir / "provenance.json").write_text("{}", encoding="utf-8")
    assert verify(proj, whdir, "android") == 0, capsys.readouterr().out
    assert verify(proj, whdir, "all") == 1     # the iOS wheels are missing
    assert "missing cryptography-50.0.2-1000-cp313-cp313-ios_13_0_arm64_iphoneos.whl" in capsys.readouterr().out


def test_verify_wheelhouse_catches_tampering(proj, capsys):
    whdir = full_wheelhouse(proj["tmp"] / "wh", targets_crypto=ANDROID)
    write_sums(whdir)
    victim = whdir / "pillow-12.3.0-1000-cp313-cp313-android_24_x86_64.whl"
    with zipfile.ZipFile(victim, "a") as zf:   # a member RECORD does not know
        zf.writestr("PIL/_evil.so", b"payload")
    assert verify(proj, whdir, "android") == 1
    out = capsys.readouterr().out
    assert "PIL/_evil.so is not in RECORD" in out
    assert "pillow-12.3.0-1000-cp313-cp313-android_24_x86_64.whl: sha256 differs from SHA256SUMS" in out


def test_verify_wheelhouse_record_hash_mismatch(proj, capsys):
    whdir = proj["tmp"] / "wh"
    good = make_wheel(proj["tmp"] / "src", "cryptography", "50.0.2", "cp313-cp313-android_24_arm64_v8a",
                      requires=CRYPTO_REQS, files={"cryptography/__init__.py": b"x = 1\n"})
    whdir.mkdir()
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(whdir / good.name, "w") as dst:
        for info in src.infolist():
            data = src.read(info)
            dst.writestr(info, b"x = 2\n" if info.filename == "cryptography/__init__.py" else data)
    rc = verify(proj, whdir, "android")
    assert rc == 1 and "cryptography/__init__.py does not match its RECORD hash" in capsys.readouterr().out


def make_site_packages(root: Path, targets, build="1000", name="cryptography", version="50.0.2"):
    for t in targets:
        di = root / t / f"{name}-{version}.dist-info"
        di.mkdir(parents=True)
        lines = ["Wheel-Version: 1.0", "Root-Is-Purelib: false"] + ([f"Build: {build}"] if build else [])
        lines.append(f"Tag: cp313-cp313-{LABEL[t]}")
        (di / "WHEEL").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_verify_installed(proj, capsys):
    root = proj["root"]
    sp = proj["tmp"] / "site-packages"
    make_site_packages(sp, ANDROID)
    make_site_packages(sp, ANDROID, name="pillow", version="12.3.0")
    whdir = full_wheelhouse(proj["tmp"] / "wh", targets_crypto=ANDROID)
    argv = ["--wheelhouse-manifest", str(root / "wheelhouse.toml"), "--verify-installed", str(sp),
            "--platform", "android"]
    assert cmw.main(argv + ["--wheelhouse", str(whdir)]) == 0, capsys.readouterr().out
    assert cmw.main(argv) == 0
    capsys.readouterr()
    # pypi.flet.dev's copy: Build 11
    other = proj["tmp"] / "site-packages-index"
    make_site_packages(other, ANDROID, build="11")
    make_site_packages(other, ANDROID, name="pillow", version="12.3.0", build=None)
    argv[3] = str(other)
    assert cmw.main(argv) == 1
    out = capsys.readouterr().out
    assert "arm64-v8a: cryptography 50.0.2 was installed from an index, not the wheelhouse (WHEEL Build 11" in out
    assert "x86_64: pillow 12.3.0 was installed from an index, not the wheelhouse (WHEEL Build (none)" in out
    # nothing built for the platform at all
    argv[3] = str(proj["tmp"] / "empty")
    assert cmw.main(argv) == 1


def test_verify_installed_skips_arches_that_were_not_built(proj):
    sp = proj["tmp"] / "sp"
    make_site_packages(sp, ["iphonesimulator.arm64", "iphonesimulator.x86_64"])
    manifest = cmw.load_wheelhouse_manifest(proj["root"] / "wheelhouse.toml")
    errors, notes = cmw.verify_installed(sp, manifest, "ios")
    assert errors == [] and any("iphoneos.arm64: no" in n for n in notes)


# --------------------------------------------------------------------------- manifest
@pytest.mark.parametrize("change, message", [
    (("schema = 1", "schema = 3"), "unsupported schema"),
    (("build_tag = 1000", "build_tag = 0"), "build_tag must be a positive integer"),
    (('targets = ["arm64-v8a", "x86_64"]', 'targets = ["arm64-v8a", "mips"]'), "targets must be distinct keys"),
    ((f'sdist_sha256 = "{PILLOW_SHA}"', 'sdist_sha256 = "abc"'), "sdist_sha256 must be 64"),
    (('requires_dist = ["flet-libjpeg', 'requires_dist = ["furo; extra == \\"docs\\"", "flet-libjpeg'),
     "only what installs without extras"),
])
def test_manifest_validation(tmp_path, change, message):
    text = MANIFEST.replace(*change)
    assert text != MANIFEST
    p = tmp_path / "m.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(cmw.ManifestError, match=message):
        cmw.load_wheelhouse_manifest(p)


def test_requirement_keys_ignore_spelling():
    a = cmw.requirement_key("flet-libjpeg (==3.0.90)")
    assert a == cmw.requirement_key("flet_libjpeg==3.0.90")
    assert cmw.requirement_key("cffi>=2.0.0 ; platform_python_implementation != 'PyPy'") == \
        cmw.requirement_key('cffi>=2.0.0; platform_python_implementation != "PyPy"')
    assert cmw.runtime_requirements(['bcrypt>=3.1.5; extra == "ssh"', "cffi>=2"]) == {cmw.requirement_key("cffi>=2")}


# --------------------------------------------------------------------------- the committed files
def test_real_manifest_matches_pins_lock_and_recipes():
    manifest = cmw.load_wheelhouse_manifest(cmw.DEFAULT_WHEELHOUSE_MANIFEST)
    project = cmw.load_project(cmw.DEFAULT_PYPROJECT)
    assert cmw.check_manifest_pins(manifest, project, MOBILE / "uv.lock") == []
    assert {w.name: w.version for w in manifest.wheels} == {"cryptography": "50.0.2", "pillow": "12.3.0"}
    assert manifest.build_tag > 11, "must outrank pypi.flet.dev's build tags"
    crypto = manifest.wheel("cryptography")
    assert set(crypto.targets) == set(ANDROID + IOS), "serious_python installs all three iOS slices"
    assert set(project.android_archs) <= set(crypto.targets)
    ctx = wh.Ctx(cmw.DEFAULT_WHEELHOUSE_MANIFEST, "android")
    for w in manifest.wheels:
        recipe = ctx.recipe_dir(w)   # version and build number agree with the manifest
        assert recipe.parent.name == "recipes"
    data = manifest.data
    assert data["toolchain"]["pip_constraints"]["cffi"] == "2.0.0"
    for plat, targets in (("android", ANDROID), ("ios", IOS)):
        assets = data["openssl"][plat]["assets"]
        assert set(assets) == set(targets)
        for asset in assets.values():
            assert re.fullmatch(r"[0-9a-f]{64}", asset["sha256"])
            assert asset["file"].startswith(f"openssl-{data['openssl']['version']}-{data['openssl'][plat]['build']}-")


def test_cryptography_requires_dist_matches_the_sdist_metadata():
    """The promised Requires-Dist is what cryptography 50.0.2's own metadata declares without extras."""
    manifest = cmw.load_wheelhouse_manifest(cmw.DEFAULT_WHEELHOUSE_MANIFEST)
    sdist_requires = ["cffi>=2.0.0 ; platform_python_implementation != 'PyPy'",
                      "typing-extensions>=4.13.2 ; python_full_version < '3.11'",
                      "bcrypt>=3.1.5 ; extra == 'ssh'"]     # PKG-INFO of cryptography-50.0.2.tar.gz
    assert cmw.runtime_requirements(manifest.wheel("cryptography").requires_dist) == \
        cmw.runtime_requirements(sdist_requires)


def _render_recipe(path: Path, sdk: str) -> dict:
    jinja2 = pytest.importorskip("jinja2")
    yaml = pytest.importorskip("yaml")
    text = jinja2.Template(path.read_text(encoding="utf-8")).render(
        sdk=sdk, sdk_version="24" if sdk == "android" else "13.0", arch="arm64", version=None,
        py_version=sys.version_info)
    return yaml.safe_load(text)


def test_recipes_render_like_forge():
    recipes = MOBILE / "ci" / "wheels" / "recipes"
    android = _render_recipe(recipes / "cryptography" / "meta.yaml", "android")
    ios = _render_recipe(recipes / "cryptography" / "meta.yaml", "iphoneos")
    assert android["package"] == {"name": "cryptography", "version": "50.0.2"}
    assert android["build"]["number"] == 1000 and android["requirements"]["host"] == ["openssl 3.5.9"]
    assert android["build"]["script_env"]["OPENSSL_STATIC"] == "1"
    assert "OPENSSL_STATIC" not in ios["build"]["script_env"]       # clang_rt.osx otherwise
    assert ios["build"]["script_env"]["OPENSSL_DIR"] == "{platlib}/opt"
    # forge's build environment starts empty: the recipe is the only way to reach rustc
    manifest = cmw.load_wheelhouse_manifest(cmw.DEFAULT_WHEELHOUSE_MANIFEST)
    assert ios["build"]["script_env"]["IPHONEOS_DEPLOYMENT_TARGET"] == manifest.data["toolchain"]["ios_deployment_target"]
    assert "IPHONEOS_DEPLOYMENT_TARGET" not in android["build"]["script_env"]
    pillow = _render_recipe(recipes / "pillow" / "meta.yaml", "android")
    assert pillow["package"]["version"] == "12.3.0" and pillow["build"]["number"] == 1000
    assert pillow["patches"] == ["setup-12.x.patch"]
    assert pillow["requirements"]["host"] == ["flet-libjpeg 3.0.90", "flet-libfreetype 2.13.3", "flet-libwebp 1.6.0"]
    assert (recipes / "pillow" / "patches" / "setup-12.x.patch").is_file()


def test_recipe_meta_reader():
    recipes = MOBILE / "ci" / "wheels" / "recipes"
    assert wh.recipe_meta(recipes / "cryptography" / "meta.yaml") == {
        "name": "cryptography", "version": "50.0.2", "openssl": "openssl 3.5.9", "number": "1000",
        "ios_deployment_target": "13.0"}
    assert wh.recipe_meta(recipes / "pillow" / "meta.yaml") == {"name": "Pillow", "version": "12.3.0", "number": "1000"}


def test_ios_recipes_must_set_the_deployment_target(tmp_path):
    """An exported IPHONEOS_DEPLOYMENT_TARGET never reaches forge's build: the recipe must set it."""
    recipes = MOBILE / "ci" / "wheels" / "recipes"
    (tmp_path / "recipes" / "cryptography").mkdir(parents=True)
    meta = (recipes / "cryptography" / "meta.yaml").read_text(encoding="utf-8")
    manifest = tmp_path / "wheelhouse.toml"
    manifest.write_text(cmw.DEFAULT_WHEELHOUSE_MANIFEST.read_text(encoding="utf-8").replace(
        '"recipes/pillow"', f'"{(recipes / "pillow").as_posix()}"'), encoding="utf-8")
    target = tmp_path / "recipes" / "cryptography" / "meta.yaml"
    for text, ok in ((meta, True),
                     (meta.replace("    IPHONEOS_DEPLOYMENT_TARGET: '13.0'\n", ""), False),
                     (meta.replace("IPHONEOS_DEPLOYMENT_TARGET: '13.0'", "IPHONEOS_DEPLOYMENT_TARGET: '12.0'"), False)):
        assert ok or text != meta
        target.write_text(text, encoding="utf-8")
        ctx = wh.Ctx(manifest, "ios")
        if ok:
            assert ctx.recipe_dir(ctx.manifest.wheel("cryptography")) == target.parent.resolve()
        else:
            with pytest.raises(wh.WheelsError, match="IPHONEOS_DEPLOYMENT_TARGET"):
                ctx.recipe_dir(ctx.manifest.wheel("cryptography"))
    # Android-only wheels need no deployment target
    ctx = wh.Ctx(manifest, "android")
    assert ctx.recipe_dir(ctx.manifest.wheel("pillow")) == (recipes / "pillow").resolve()


def test_pillow_patch_is_upstream_verbatim():
    """recipes/pillow/patches/setup-12.x.patch = mobile-forge@0fadbfab's file (git blob id)."""
    data = (MOBILE / "ci" / "wheels" / "recipes" / "pillow" / "patches" / "setup-12.x.patch").read_bytes()
    data = data.replace(b"\r\n", b"\n")      # a Windows checkout may add CRs
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    assert blob == "62279a3576b8e1ea0997eb1fe89c7c7e337729eb"
    assert (MOBILE / "ci" / "wheels" / "recipes" / "LICENSE.mobile-forge").read_text(encoding="utf-8").startswith(
        "Copyright (c) 2023 Russell Keith-Magee.")


BUILD_SCRIPT = MOBILE / "ci" / "wheels" / "build_wheels.sh"


def test_build_script_keeps_lf_and_names_the_helpers():
    script = BUILD_SCRIPT.read_bytes()
    assert script.startswith(b"#!/usr/bin/env bash\n") and b"\r" not in script
    for needle in (b"wheels_py\" env", b"fetch --platform", b"stage-openssl", b"source ./setup.sh", b"collect --platform"):
        assert needle in script


def test_build_script_backslashes_are_only_continuations_or_quotes():
    """Outside comments, a backslash ends a line (continuation) or escapes a double quote.

    `\\n` written for a line break is a word `n` to bash (`bash -n` accepts it); that once gave
    stage-openssl a package named "n" on every build.
    """
    bad = []
    for no, line in enumerate(BUILD_SCRIPT.read_text(encoding="utf-8").split("\n"), 1):
        if line.lstrip().startswith("#"):
            continue
        for m in re.finditer(r"\\(.?)", line):
            if m.group(1) not in ('"', ""):
                bad.append(f"{no}: {line.strip()}")
    assert bad == []


def _bash() -> str:
    """bash for the stubbed build run: Git Bash on Windows (never WSL's launcher), else PATH's."""
    candidates = []
    if os.name == "nt":
        for base in (os.environ.get("ProgramW6432"), os.environ.get("ProgramFiles")):
            if base:
                candidates.append(str(Path(base) / "Git" / "bin" / "bash.exe"))
    found = shutil.which("bash")
    if found and not (os.name == "nt" and re.search(r"[\\/](system32|windowsapps)[\\/]", found, re.IGNORECASE)):
        candidates.append(found)
    for c in candidates:
        if Path(c).is_file():
            return c
    pytest.skip("no bash")


def _arg(value: str) -> str:
    """A logged argument with paths in one spelling (C:/x and C:\\x alike on Windows)."""
    if re.match(r"^([A-Za-z]:)?[\\/]", value):
        return os.path.normcase(os.path.normpath(value))
    return value


# One tab-separated line per call; Git Bash's POSIX paths (/c/..., /tmp/...) logged as C:/... .
STUB_LOG = """\
log() {
  local IFS=$'\\t' a out=()
  for a in "$@"; do
    case "$a" in /*) if command -v cygpath > /dev/null; then a="$(cygpath -m "$a")"; fi ;; esac
    out+=("$a")
  done
  printf '%s\\n' "${out[*]}" >> "$STUB_CALLS"
}
"""
STUBS = {
    # build_wheels.sh runs wheels.py through `uv run --no-project python`: the real wheels.py for the
    # pinned values (env, recipe), only logged for the network step (fetch).
    "shim/uv": "#!/usr/bin/env bash\nset -euo pipefail\n" + STUB_LOG + """\
[ "${1:-} ${2:-} ${3:-}" = "run --no-project python" ] || { echo "stub uv: unexpected: $*" >&2; exit 97; }
shift 3
log uv-python "$@"
case "${2:-}" in
  env|recipe|packages) "$STUB_REAL_PY" "$@" | tr -d '\\r' ;;
esac
""",
    "shim/git": "#!/usr/bin/env bash\nset -euo pipefail\n" + STUB_LOG + """\
log git "$@"
if [ "$1" = init ]; then mkdir -p "$3"; exit 0; fi
[ "$1" = -C ] || exit 98
dir="$2"
shift 2
case "$1" in
  rev-parse) echo "$FORGE_REV" ;;
  checkout)
    mkdir -p "$dir/.ci"
    cp "$STUB_DIR/forge/pyproject.toml" "$dir/pyproject.toml"
    cp "$STUB_DIR/forge/install_ndk.sh" "$dir/.ci/install_ndk.sh"
    cp "$STUB_DIR/forge/setup.sh" "$dir/setup.sh" ;;
esac
""",
    "forge/install_ndk.sh": STUB_LOG + 'log install_ndk.sh "$@"\n',
    # sourced by build_wheels.sh like forge's: venv with python + forge, support tree, xcframework headers
    "forge/setup.sh": """\
printf 'setup.sh\\t%s\\t%s\\n' "$1" "$2" >> "$STUB_CALLS"
mkdir -p "$PWD/.venv/bin"
cp "$STUB_DIR/venv/python" "$STUB_DIR/venv/forge" "$PWD/.venv/bin/"
chmod +x "$PWD/.venv/bin/python" "$PWD/.venv/bin/forge"
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$PWD/.venv/bin:$PATH"
export MOBILE_FORGE_ANDROID_SUPPORT_PATH="$PWD/support" MOBILE_FORGE_IOS_SUPPORT_PATH="$PWD/support"
if [ "$2" = iOS ]; then
  for slice in ios-arm64 ios-arm64_x86_64-simulator; do
    mkdir -p "$PWD/support/support/$PY_SHORT/iOS/Python.xcframework/$slice/include/python$PY_SHORT"
    echo '/* Python.h */' > "$PWD/support/support/$PY_SHORT/iOS/Python.xcframework/$slice/include/python$PY_SHORT/Python.h"
  done
fi
""",
    "venv/python": "#!/usr/bin/env bash\n" + STUB_LOG + """\
if [ "${1:-}" = -c ]; then echo "$PY_FULL"; exit 0; fi
log python "$@"
""",
    "venv/forge": "#!/usr/bin/env bash\n" + STUB_LOG + 'log forge "$@"\n',
}


@pytest.mark.parametrize("platform, packages", [
    ("android", ["cryptography", "pillow"]),
    ("android", ["pillow"]),             # a Pillow-only rebuild (cryptography came from the cache)
    ("ios", ["cryptography"]),
])
def test_build_script_runs_end_to_end_with_stubbed_tools(tmp_path, platform, packages):
    """build_wheels.sh with git, uv, forge's setup.sh, the forge venv and forge itself stubbed:
    every wheels.py and forge call gets exactly the expected arguments."""
    bash = _bash()
    manifest = cmw.load_wheelhouse_manifest(cmw.DEFAULT_WHEELHOUSE_MANIFEST).data
    tc = manifest["toolchain"]
    stubs, shim, tmp = tmp_path / "stubs", tmp_path / "stubs" / "shim", tmp_path / "runner"
    for rel, text in STUBS.items():
        p = stubs / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode())
        p.chmod(0o755)
    (stubs / "forge" / "pyproject.toml").write_bytes(
        f'[project]\ndependencies = [\n    "crossenv @ git+{tc["crossenv"]["repo"]}@flet",\n]\n'.encode())
    sdk = tmp_path / "sdk"
    clang = sdk / "ndk" / tc["android_ndk"] / "toolchains" / "llvm" / "prebuilt" / "linux-x86_64" / "bin"
    clang.mkdir(parents=True)
    (clang / "aarch64-linux-android24-clang").write_bytes(b"")
    tmp.mkdir()
    forge, cache, calls = tmp / "forge", tmp / "wheelcache", tmp_path / "calls.log"
    env = {k: v for k, v in os.environ.items() if k not in (
        "UV_PYTHON", "VIRTUAL_ENV", "PIP_CONSTRAINT", "NDK_HOME", "GITHUB_ACTIONS", "PYTHONPATH",
        "MOBILE_FORGE_ANDROID_SUPPORT_PATH", "MOBILE_FORGE_IOS_SUPPORT_PATH")}
    env.update({"STUB_CALLS": calls.as_posix(), "STUB_DIR": stubs.as_posix(), "STUB_SHIM": shim.as_posix(),
                "STUB_REAL_PY": Path(sys.executable).as_posix(), "RUNNER_TEMP": tmp.as_posix(),
                "FORGE_DIR": forge.as_posix(), "WHEELCACHE": cache.as_posix(), "ANDROID_HOME": sdk.as_posix()})
    # Put the stubs first on bash's own PATH (Git Bash's launcher prepends its git otherwise).
    launcher = ('shim="$STUB_SHIM"; if command -v cygpath > /dev/null; then shim="$(cygpath -u "$shim")"; fi; '
                'PATH="$shim:$PATH"; export PATH; exec "$BASH" "$0" "$@"')
    run = subprocess.run([bash, "-c", launcher, BUILD_SCRIPT.as_posix(), platform, *packages], env=env,
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]

    host = {"android": "android", "ios": "iOS"}[platform]
    w = _arg(str(MOBILE / "ci" / "wheels" / "wheels.py"))
    f = _arg(str(forge))

    def recipe(pkg):
        return _arg(str((MOBILE / "ci" / "wheels" / "recipes" / pkg).resolve()))

    expected = [
        ["uv-python", w, "env", "--platform", platform, "--constraints-out", _arg(str(tmp / "mobile-wheels-constraints.txt"))],
        *(["uv-python", w, "recipe", "--platform", platform, p] for p in packages),
        ["git", "init", "-q", f],
        ["git", "-C", f, "remote", "add", "origin", tc["mobile_forge"]["repo"]],
        ["git", "-C", f, "fetch", "-q", "--depth", "1", "origin", tc["mobile_forge"]["rev"]],
        ["git", "-C", f, "checkout", "-q", "--detach", "FETCH_HEAD"],
        ["git", "-C", f, "rev-parse", "HEAD"],
        ["uv-python", w, "fetch", "--platform", platform, "--dest", _arg(str(forge / "downloads")), *packages],
        *([["install_ndk.sh", tc["android_ndk"]]] if platform == "android" else []),
        ["setup.sh", tc["python_version"], host],
        ["python", w, "stage-openssl", "--platform", platform, "--downloads", _arg(str(forge / "downloads")),
         "--forge", f, *packages],
        *(["forge", host, recipe(p)] for p in packages),
        ["python", w, "collect", "--platform", platform, "--dist", _arg(str(forge / "dist")), "--logs",
         _arg(str(forge / "logs")), "--cache", _arg(str(cache)), *packages],
    ]
    got = [[_arg(a) for a in line.split("\t")] for line in calls.read_text(encoding="utf-8").splitlines()]
    assert got == expected
    pyproject = (forge / "pyproject.toml").read_text(encoding="utf-8")
    assert f'"crossenv @ git+{tc["crossenv"]["repo"]}@{tc["crossenv"]["rev"]}"' in pyproject
    assert "maturin==" in (tmp / "mobile-wheels-constraints.txt").read_text(encoding="utf-8")
    if platform == "ios":
        py_short = ".".join(tc["python_version"].split(".")[:2])
        xcf = forge / "support" / "support" / py_short / "iOS" / "Python.xcframework"
        for slice_dir, name in (("ios-arm64", "arm64-iphoneos"), ("ios-arm64_x86_64-simulator", "arm64-iphonesimulator"),
                                ("ios-arm64_x86_64-simulator", "x86_64-iphonesimulator")):
            assert (xcf / slice_dir / "include" / name / "Python.h").is_file()


# --------------------------------------------------------------------------- wheels.py
def make_elf(machine=183, align=0x4000, needed=("libc.so",), extra=b"", elf_class=2) -> bytes:
    strtab = b"\0" + b"".join(n.encode() + b"\0" for n in needed)
    offsets, pos = [], 1
    for n in needed:
        offsets.append(pos)
        pos += len(n) + 1
    dynstr_off = 64 + 2 * 56
    pad = b"\0" * ((-(dynstr_off + len(strtab))) % 8)
    dyn_off = dynstr_off + len(strtab) + len(pad)
    dyn = b"".join(struct.pack("<qQ", 1, o) for o in offsets) + struct.pack("<qQ", 5, dynstr_off) + struct.pack("<qQ", 0, 0)
    body = strtab + pad + dyn + extra
    total = dynstr_off + len(body)
    ehdr = b"\x7fELF" + bytes([elf_class, 1, 1, 0]) + b"\0" * 8 + struct.pack(
        "<HHIQQQIHHHHHH", 3, machine, 1, 0, 64, 0, 0, 64, 56, 2, 64, 0, 0)
    load = struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, total, total, align)
    dynamic = struct.pack("<IIQQQQQQ", 2, 6, dyn_off, dyn_off, dyn_off, len(dyn), len(dyn), 8)
    return ehdr + load + dynamic + body


ARM64, X86_64 = 0x0100000C, 0x01000007
LC_VERSION_MIN_MACOSX, LC_VERSION_MIN_IPHONEOS = 0x24, 0x25


def make_macho(cputype=ARM64, platform=2, minos=(13, 0, 0), dylibs=("@rpath/Python.framework/Python",
               "/usr/lib/libSystem.B.dylib"), extra=b"", version_min=None, both=False) -> bytes:
    """A thin Mach-O dylib. ``version_min`` (a LC_VERSION_MIN_* id) replaces LC_BUILD_VERSION, as ld
    writes it for deployment targets below iOS 12; ``both`` writes the two."""
    v = (minos[0] << 16) | (minos[1] << 8) | minos[2]
    plat_cmds = []
    if version_min is None or both:
        plat_cmds.append(struct.pack("<IIIIII", 0x32, 24, platform, v, 0, 0))
    if version_min is not None:
        plat_cmds.append(struct.pack("<IIII", version_min, 16, v, 0))
    cmds = b"".join(plat_cmds)
    for d in dylibs:
        name = d.encode() + b"\0"
        size = 24 + len(name)
        size += (-size) % 8
        cmd = struct.pack("<IIIIII", 0xC, size, 24, 0, 0x10000, 0x10000) + name
        cmds += cmd + b"\0" * (size - len(cmd))
    hdr = struct.pack("<IiiIIIII", 0xFEEDFACF, cputype, 0, 6, len(plat_cmds) + len(dylibs), len(cmds), 0, 0)
    return hdr + cmds + extra


OPENSSL_TEXT = b"\0OpenSSL 3.5.9 30 Sep 2026\0"


def test_parse_elf_reads_machine_alignment_and_needed():
    elf = wh.parse_elf(make_elf(needed=("libpython3.13.so", "libdl.so", "libc.so")))
    assert elf == {"class": 64, "machine": 183, "load_aligns": [0x4000],
                   "needed": ["libpython3.13.so", "libdl.so", "libc.so"]}
    with pytest.raises(ValueError):
        wh.parse_elf(b"MZ\0\0")


def test_android_binary_checks():
    name = "cryptography/hazmat/bindings/_rust.abi3.so"
    good = make_elf(needed=("libpython3.13.so", "libdl.so", "libc.so"), extra=OPENSSL_TEXT)
    assert wh.check_android_binary(name, good, "arm64-v8a", "cryptography", "3.5.9") == []
    errs = wh.check_android_binary(name, good, "x86_64", "cryptography", "3.5.9")
    assert any("ELF machine 183 is not x86_64" in e for e in errs)
    assert any("LOAD alignment" in e for e in wh.check_android_binary(
        name, make_elf(align=0x1000, extra=OPENSSL_TEXT), "arm64-v8a", "cryptography", "3.5.9"))
    errs = wh.check_android_binary(name, make_elf(needed=("libpython3.so", "libcrypto_python.so", "libz.so")),
                                   "arm64-v8a", "cryptography", "3.5.9")
    text = "\n".join(errs)
    assert "libpython3.so" in text and "libcrypto_python.so" in text and "libz.so" in text
    assert "no 'OpenSSL 3.5.9' version string" in text
    old = make_elf(needed=("libc.so",), extra=b"OpenSSL 3.0.21 1 Jan 2026")
    assert any("OpenSSL 3.5.9" in e for e in wh.check_android_binary(name, old, "arm64-v8a", "cryptography", "3.5.9"))
    imaging = "PIL/_imaging.cpython-313-aarch64-linux-android.so"
    assert wh.check_android_binary(imaging, make_elf(needed=("libjpeg.so", "libz.so")), "arm64-v8a", "pillow", "3.5.9") == []
    assert any("libjpeg.so" in e for e in wh.check_android_binary(imaging, make_elf(), "arm64-v8a", "pillow", "3.5.9"))


def test_ios_binary_checks():
    name = "cryptography/hazmat/bindings/_rust.abi3.so"
    good = make_macho(extra=OPENSSL_TEXT)
    assert wh.parse_macho(good)["builds"] == [(2, (13, 0, 0))] and wh.parse_macho(good)["version_mins"] == []
    assert wh.check_ios_binary(name, good, "iphoneos.arm64", "cryptography", "3.5.9", (13, 0)) == []
    sim = make_macho(platform=7, cputype=X86_64, extra=OPENSSL_TEXT)
    assert wh.check_ios_binary(name, sim, "iphonesimulator.x86_64", "cryptography", "3.5.9", (13, 0)) == []
    errs = "\n".join(wh.check_ios_binary(name, good, "iphonesimulator.arm64", "cryptography", "3.5.9", (13, 0)))
    assert "platform IOS (LC_BUILD_VERSION) is not IOSSIMULATOR" in errs
    errs = "\n".join(wh.check_ios_binary(name, make_macho(minos=(15, 0, 0), dylibs=(
        "@rpath/Python.framework/Python", "/opt/homebrew/lib/libssl.3.dylib")), "iphoneos.arm64", "cryptography",
        "3.5.9", (13, 0)))
    assert "minos 15.0.0 (LC_BUILD_VERSION) is above 13.0" in errs and "libssl.3.dylib" in errs
    assert "OpenSSL 3.5.9" in errs
    fat = struct.pack(">II", 0xCAFEBABE, 2) + b"\0" * 64
    assert "universal" in "\n".join(wh.check_ios_binary(name, fat, "iphoneos.arm64", "cryptography", "3.5.9", (13, 0)))


# The load commands forge's Rust builds really carry (pypi.flet.dev's cryptography-43.0.1-10 and
# pydantic_core-2.47.0-4 cp313 iOS wheels, read with wheels.parse_macho): without an
# IPHONEOS_DEPLOYMENT_TARGET in the build environment, rustc links for its default iOS 10.0, so
# ld writes LC_VERSION_MIN_IPHONEOS on the device and x86_64 simulator slices; the arm64 simulator
# slice gets LC_BUILD_VERSION IOSSIMULATOR 14.0 (LLVM's floor; PyPI's ios_13_0 Pillow too).
FORGE_DYLIBS = ("/usr/lib/libiconv.2.dylib", "/usr/lib/libSystem.B.dylib", "@rpath/Python.framework/Python")
FORGE_RUST_SLICES = {
    "iphoneos.arm64": make_macho(ARM64, minos=(10, 0, 0), version_min=LC_VERSION_MIN_IPHONEOS, dylibs=FORGE_DYLIBS,
                                 extra=OPENSSL_TEXT),
    "iphonesimulator.arm64": make_macho(ARM64, platform=7, minos=(14, 0, 0), dylibs=FORGE_DYLIBS, extra=OPENSSL_TEXT),
    "iphonesimulator.x86_64": make_macho(X86_64, minos=(10, 0, 0), version_min=LC_VERSION_MIN_IPHONEOS,
                                         dylibs=FORGE_DYLIBS, extra=OPENSSL_TEXT),
}
# With the recipe's IPHONEOS_DEPLOYMENT_TARGET=13.0: LC_BUILD_VERSION 13.0, the arm64 simulator still 14.0.
RECIPE_RUST_SLICES = {
    "iphoneos.arm64": make_macho(ARM64, platform=2, minos=(13, 0, 0), dylibs=FORGE_DYLIBS, extra=OPENSSL_TEXT),
    "iphonesimulator.arm64": make_macho(ARM64, platform=7, minos=(14, 0, 0), dylibs=FORGE_DYLIBS, extra=OPENSSL_TEXT),
    "iphonesimulator.x86_64": make_macho(X86_64, platform=7, minos=(13, 0, 0), dylibs=FORGE_DYLIBS, extra=OPENSSL_TEXT),
}


@pytest.mark.parametrize("slices", [FORGE_RUST_SLICES, RECIPE_RUST_SLICES], ids=["forge-default", "recipe-13.0"])
def test_ios_binary_checks_accept_what_forge_builds(slices):
    name = "cryptography/hazmat/bindings/_rust.abi3.so"
    for target, data in slices.items():
        assert wh.check_ios_binary(name, data, target, "cryptography", "3.5.9", (13, 0)) == [], target


def test_ios_binary_checks_reject_the_wrong_slice_platform_or_minos():
    name = "cryptography/hazmat/bindings/_rust.abi3.so"

    def errs(data, target):
        return "\n".join(wh.check_ios_binary(name, data, target, "cryptography", "3.5.9", (13, 0)))

    device, sim_arm, sim_x86 = (FORGE_RUST_SLICES[t] for t in IOS)
    # LC_VERSION_MIN_IPHONEOS on arm64 is a device binary, never an arm64 simulator one
    assert "platform IOS (LC_VERSION_MIN_IPHONEOS) is not IOSSIMULATOR" in errs(device, "iphonesimulator.arm64")
    assert "platform IOSSIMULATOR (LC_BUILD_VERSION) is not IOS" in errs(sim_arm, "iphoneos.arm64")
    assert "cputype 0x1000007 is not iphoneos.arm64" in errs(sim_x86, "iphoneos.arm64")
    # 14.0 is allowed only on the arm64 simulator slice
    assert "minos 14.0.0 (LC_BUILD_VERSION) is above 13.0" in errs(
        make_macho(X86_64, platform=7, minos=(14, 0, 0), extra=OPENSSL_TEXT), "iphonesimulator.x86_64")
    assert "minos 14.0.0 (LC_VERSION_MIN_IPHONEOS) is above 13.0" in errs(
        make_macho(ARM64, minos=(14, 0, 0), version_min=LC_VERSION_MIN_IPHONEOS, extra=OPENSSL_TEXT), "iphoneos.arm64")
    assert "minos 15.0.0 (LC_BUILD_VERSION) is above 14.0" in errs(
        make_macho(ARM64, platform=7, minos=(15, 0, 0), extra=OPENSSL_TEXT), "iphonesimulator.arm64")
    # a macOS binary, two platform commands, none at all
    assert "platform MACOS (LC_VERSION_MIN_MACOSX) is not IOS" in errs(
        make_macho(ARM64, minos=(11, 0, 0), version_min=LC_VERSION_MIN_MACOSX, extra=OPENSSL_TEXT), "iphoneos.arm64")
    assert "2 platform load commands" in errs(
        make_macho(ARM64, minos=(13, 0, 0), version_min=LC_VERSION_MIN_IPHONEOS, both=True, extra=OPENSSL_TEXT),
        "iphoneos.arm64")
    hdr_only = struct.pack("<IiiIIIII", 0xFEEDFACF, ARM64, 0, 6, 0, 0, 0, 0) + OPENSSL_TEXT
    assert "0 platform load commands" in errs(hdr_only, "iphoneos.arm64")


def test_verify_native_accepts_forge_shaped_ios_wheels(proj, tmp_path, capsys):
    so = "cryptography/hazmat/bindings/_rust.abi3.so"
    for target, data in FORGE_RUST_SLICES.items():
        make_wheel(tmp_path / "wh", "cryptography", "50.0.2", f"cp313-cp313-{LABEL[target]}", requires=CRYPTO_REQS,
                   files={so: data})
    argv = ["--manifest", str(proj["root"] / "wheelhouse.toml"), "verify-native", "--platform", "ios",
            str(tmp_path / "wh")]
    assert wh.main(argv) == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "IOS 10.0.0 (LC_VERSION_MIN_IPHONEOS)" in out and "IOSSIMULATOR 14.0.0 (LC_BUILD_VERSION)" in out


def test_verify_native_wheel(tmp_path):
    so = "cryptography/hazmat/bindings/_rust.abi3.so"
    whl = make_wheel(tmp_path, "cryptography", "50.0.2", "cp313-cp313-android_24_x86_64", requires=CRYPTO_REQS,
                     files={so: make_elf(machine=62, needed=("libpython3.13.so", "libc.so"), extra=OPENSSL_TEXT)})
    errors, notes = wh.verify_native_wheel(whl, "cryptography", "x86_64", "3.5.9", (13, 0))
    assert errors == [] and notes
    pil = make_wheel(tmp_path, "pillow", "12.3.0", "cp313-cp313-android_24_arm64_v8a",
                     files={"PIL/_imaging.cpython-313-aarch64-linux-android.so": make_elf(needed=("libjpeg.so",))})
    errors, _ = wh.verify_native_wheel(pil, "pillow", "arm64-v8a", "3.5.9", (13, 0))
    assert errors == [f"{pil.name}: no PIL/_imagingft*.so", f"{pil.name}: no PIL/_webp*.so"]


def selftest_report(**overrides) -> dict:
    report = {
        "suite": "smoke", "ok": True, "platform": "android", "strict": True,
        "checks": [
            {"name": "fernet", "status": "pass", "detail": {"version": "50.0.2", "openssl": "OpenSSL 3.5.9 30 Sep 2026"}},
            {"name": "pillow", "status": "pass", "detail": {
                "version": "12.3.0", "features": {"jpg": True, "zlib": True, "webp": True, "freetype2": True},
                "roundtrip": {"JPEG": [16, 16], "PNG": [16, 16], "WEBP": [16, 16]}, "font": "FreeTypeFont"}},
        ],
    }
    report.update(overrides)
    return report


def test_selftest_assertion():
    assert wh.check_selftest_report(selftest_report(), "android", "50.0.2", "3.5.9", "12.3.0") == []
    errs = wh.check_selftest_report(selftest_report(suite="e2e", platform="ios"), "android", "50.0.2", "3.5.9", "12.3.0")
    assert any("suite is 'e2e'" in e for e in errs) and any("platform is 'ios'" in e for e in errs)
    report = selftest_report()
    report["checks"][0]["detail"]["openssl"] = "OpenSSL 3.0.21 27 Jan 2026"
    report["checks"][1]["detail"]["features"]["webp"] = False
    report["checks"][1]["detail"]["font"] = "ImageFont"
    errs = "\n".join(wh.check_selftest_report(report, "android", "50.0.2", "3.5.9", "12.3.0"))
    assert "expected OpenSSL 3.5.9" in errs and "['webp']" in errs and "FreeTypeFont" in errs
    report = selftest_report()
    del report["checks"][1]
    assert wh.check_selftest_report(report, "android", "50.0.2", "3.5.9", "12.3.0") == [
        "check 'pillow' is missing from the report"]


def test_assert_selftest_command_reads_pins(proj, tmp_path, capsys):
    root = proj["root"]
    report = tmp_path / "selftest-smoke.json"
    report.write_text(json.dumps(selftest_report()), encoding="utf-8")
    base = ["--manifest", str(root / "wheelhouse.toml"), "--pyproject", str(root / "pyproject.toml")]
    assert wh.main(base + ["assert-selftest", "--platform", "android", str(report)]) == 0
    (root / "pyproject.toml").write_text(PYPROJECT.replace("pillow==12.3.0", "pillow==12.4.0"), encoding="utf-8")
    assert wh.main(base + ["assert-selftest", "--platform", "android", str(report)]) == 1
    assert "pyproject pins pillow 12.4.0" in capsys.readouterr().err


def wheels_marker(report: dict) -> str:
    """The line selftest.py prints for the smoke suite (compact JSON, ASCII)."""
    payload = {k: report[k] for k in ("suite", "platform", "strict")}
    payload["checks"] = [c for c in report["checks"] if c["name"] in ("fernet", "pillow")]
    return "GLOSSARION_WHEELS " + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def logcat(*messages: str) -> str:
    """`adb logcat -v threadtime -s flet.python` lines."""
    return "".join(f"10-08 12:00:{i:02d}.123  4321  4388 I flet.python: {m}\r\n" for i, m in enumerate(messages))


def test_assert_selftest_reads_the_marker_from_logcat(proj, tmp_path, capsys):
    """Release-mode APKs are not debuggable, so CI reads the smoke suite's details from logcat."""
    root = proj["root"]
    base = ["--manifest", str(root / "wheelhouse.toml"), "--pyproject", str(root / "pyproject.toml"),
            "assert-selftest", "--platform", "android"]
    log = tmp_path / "logcat_flet_python.txt"
    old = selftest_report()
    old["checks"][0]["detail"]["openssl"] = "OpenSSL 3.0.21 27 Jan 2026"
    log.write_text(logcat("GLOSSARION_READY", wheels_marker(old), 'GLOSSARION_SELFTEST PASS {"suite":"smoke"}',
                          wheels_marker(selftest_report()), 'GLOSSARION_SELFTEST PASS {"suite":"smoke"}',
                          'GLOSSARION_SELFTEST PASS {"suite":"e2e"}'), encoding="utf-8")
    assert wh.main(base + [str(log)]) == 0, capsys.readouterr()        # the last smoke marker counts
    assert "GLOSSARION_WHEELS line, 2 found" in capsys.readouterr().out
    log.write_text(logcat(wheels_marker(old)), encoding="utf-8")
    assert wh.main(base + [str(log)]) == 1
    assert "expected OpenSSL 3.5.9" in capsys.readouterr().out
    report, source = wh.load_selftest_evidence([log])
    assert wh.check_selftest_report(report, "android", "50.0.2", "3.0.21", "12.3.0") == [] and str(log) in source


def test_assert_selftest_falls_back_through_its_inputs(proj, tmp_path, capsys):
    root = proj["root"]
    base = ["--manifest", str(root / "wheelhouse.toml"), "--pyproject", str(root / "pyproject.toml"),
            "assert-selftest", "--platform", "android"]
    missing, tag_log, full_log = tmp_path / "selftest-smoke.json", tmp_path / "tag.txt", tmp_path / "full.txt"
    tag_log.write_text(logcat("GLOSSARION_READY"), encoding="utf-8")
    full_log.write_text(logcat("noise", wheels_marker(selftest_report())), encoding="utf-8")
    assert wh.main(base + [str(missing), str(tag_log), str(full_log)]) == 0
    assert f"{full_log} (GLOSSARION_WHEELS line, 1 found)" in capsys.readouterr().out
    # nothing usable: a clear failure, not a traceback
    cut = wheels_marker(selftest_report())[:120]
    full_log.write_text(logcat(cut), encoding="utf-8")
    assert wh.main(base + [str(missing), str(tag_log), str(full_log)]) == 1
    out = capsys.readouterr().out
    assert f"{missing}: not found" in out and f"{tag_log}: no GLOSSARION_WHEELS line" in out
    assert "none valid JSON" in out and "no self-test report or GLOSSARION_WHEELS line: FAIL" in out
    # a JSON report is used as it is, even when a log follows
    missing.write_text(json.dumps(selftest_report(suite="e2e")), encoding="utf-8")
    assert wh.main(base + [str(missing), str(full_log)]) == 1
    assert "suite is 'e2e'" in capsys.readouterr().out


def test_wheels_marker_needs_its_own_token():
    assert wh._wheels_payloads("X_GLOSSARION_WHEELS {}\nGLOSSARION_WHEELSX {}\n") == []
    assert wh._wheels_payloads("I flet.python: GLOSSARION_WHEELS {\"a\":1}  \r\n") == ['{"a":1}']


def _tar_bytes(members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_extract_static_openssl_keeps_only_headers_and_static_libs(tmp_path):
    tarball = tmp_path / "ossl.tar.gz"
    tarball.write_bytes(_tar_bytes({
        "include/openssl/opensslv.h": b'# define OPENSSL_VERSION_STR "3.5.9"\n',
        "include/openssl/ssl.h": b"/* ssl */\n",
        "lib/libcrypto.a": b"!<arch>\n", "lib/libssl.a": b"!<arch>\n",
        "lib/libcrypto_python.so": b"\x7fELF", "lib/ossl-modules/legacy.so": b"\x7fELF", "bin/openssl": b"\x7fELF",
    }))
    dest = tmp_path / "out"
    wh.extract_static_openssl(tarball, dest, "3.5.9")
    files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert files == ["include/openssl/opensslv.h", "include/openssl/ssl.h", "lib/libcrypto.a", "lib/libssl.a"]
    with pytest.raises(wh.WheelsError, match="not OpenSSL 3.5.8"):
        wh.extract_static_openssl(tarball, tmp_path / "out2", "3.5.8")


def test_rewrite_versions(tmp_path):
    v = tmp_path / "VERSIONS"
    v.write_text("Python version: 3.13.15\nMin iOS version: 13.0\n---\nBZip2: 1.0.8-2\nOpenSSL: 3.0.18-1\n", encoding="utf-8")
    wh.rewrite_versions(v, "3.5.9", "1")
    assert "OpenSSL: 3.5.9-1\n" in v.read_text(encoding="utf-8") and "BZip2: 1.0.8-2" in v.read_text(encoding="utf-8")
    v.write_text("bzip2: 1.0.8-1\n", encoding="utf-8")
    with pytest.raises(wh.WheelsError):
        wh.rewrite_versions(v, "3.5.9", "0")


def test_download_verified_rejects_wrong_hash(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    url = src.resolve().as_uri()
    dest = tmp_path / "dl" / "file.bin"
    with pytest.raises(wh.WheelsError, match="does not match the pinned"):
        wh.download_verified(url, dest, "0" * 64, attempts=1)
    assert not dest.exists() and not dest.with_name("file.bin.part").exists()
    wh.download_verified(url, dest, hashlib.sha256(b"payload").hexdigest(), attempts=1)
    assert dest.read_bytes() == b"payload"


def test_collect_and_assemble(proj, tmp_path, monkeypatch):
    root = proj["root"]
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    dist = tmp_path / "dist"
    full_wheelhouse(dist)
    make_wheel(dist, "openssl", "3.5.9", "py3-none-android_24_arm64_v8a", build="0")      # forge's dependency wheels
    make_wheel(dist, "flet-libjpeg", "3.0.90", "py3-none-android_24_arm64_v8a", build="1")
    cache, out = tmp_path / "cache", tmp_path / "wheelhouse"
    base = ["--manifest", str(root / "wheelhouse.toml"), "--pyproject", str(root / "pyproject.toml")]
    assert wh.main(base + ["collect", "--platform", "android", "--dist", str(dist), "--cache", str(cache)]) == 0
    assert sorted(p.name for p in (cache / "pillow").iterdir()) == [
        "build-info.json", "pillow-12.3.0-1000-cp313-cp313-android_24_arm64_v8a.whl",
        "pillow-12.3.0-1000-cp313-cp313-android_24_x86_64.whl"]
    assert len(list((cache / "cryptography").glob("*.whl"))) == 2      # only the Android slices
    assert wh.main(base + ["assemble", "--platform", "android", "--cache", str(cache), "--out", str(out)]) == 0
    assert wh.main(base + ["provenance", "--platform", "android", "--wheelhouse", str(out), "--cache", str(cache),
                           "--built", "cryptography"]) == 0
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert {e["package"]: e["built_in_this_run"] for e in prov["wheels"]} == {"cryptography": True, "pillow": False}
    assert verify(proj, out, "android") == 0
    # a missing slice fails collect
    (dist / "cryptography-50.0.2-1000-cp313-cp313-android_24_x86_64.whl").unlink()
    assert wh.main(base + ["collect", "--platform", "android", "--dist", str(dist), "--cache", str(cache)]) == 1


def test_openssl_source_is_proven_from_the_forge_log():
    label = "android_24_arm64_v8a"
    ok = ("Looking in links: /tmp/forge/dist\n"
          "Processing /tmp/forge/dist/openssl-3.5.9-0-py3-none-android_24_arm64_v8a.whl\n"
          "Installing collected packages: openssl\n")
    assert wh.check_openssl_source(ok, "3.5.9", "0", label) is None
    assert "does not show" in wh.check_openssl_source(ok.replace("3.5.9-0", "3.0.21-1"), "3.5.9", "0", label)
    hijack = ok + "Collecting openssl==3.5.9\n  Downloading openssl-3.5.9-9999-py3-none-android_24_arm64_v8a.whl (1 MB)\n"
    assert "from an index" in wh.check_openssl_source(hijack, "3.5.9", "0", label)


def test_collect_requires_the_staged_openssl_in_the_forge_log(proj, tmp_path):
    recipes = MOBILE / "ci" / "wheels" / "recipes"
    manifest = proj["root"] / "wheelhouse.toml"
    manifest.write_text(MANIFEST.replace('recipe = "recipes/', f'recipe = "{recipes.as_posix()}/'), encoding="utf-8")
    dist, logs = tmp_path / "dist", tmp_path / "logs"
    full_wheelhouse(dist, targets_crypto=ANDROID)
    logs.mkdir()
    for t in ANDROID:
        (logs / f"cryptography-50.0.2-cp313-{LABEL[t]}.log").write_text(
            f"Processing {dist.as_posix()}/openssl-3.5.9-0-py3-none-{LABEL[t]}.whl\n", encoding="utf-8")
    argv = ["--manifest", str(manifest), "--pyproject", str(proj["root"] / "pyproject.toml"), "collect",
            "--platform", "android", "--dist", str(dist), "--cache", str(tmp_path / "cache"), "--logs", str(logs)]
    assert wh.main(argv) == 0
    (logs / "cryptography-50.0.2-cp313-android_24_x86_64.log").write_text(
        "Downloading openssl-3.5.9-0-py3-none-android_24_x86_64.whl\n", encoding="utf-8")
    assert wh.main(argv) == 1


def test_env_exports(proj, capsys):
    root = proj["root"]
    base = ["--manifest", str(cmw.DEFAULT_WHEELHOUSE_MANIFEST)]
    assert wh.main(base + ["env", "--platform", "android"]) == 0
    out = capsys.readouterr().out
    assert "export PY_FULL=3.13.15" in out and "export ANDROID_ABIS='arm64-v8a x86_64'" in out
    assert "export RUST_TARGETS=aarch64-linux-android,x86_64-linux-android" in out
    assert wh.main(base + ["env", "--platform", "ios", "--format", "github"]) == 0
    out = capsys.readouterr().out
    # forge never passes the outer environment to the build: the deployment target is the recipe's
    assert "forge_host=iOS" in out and "iphoneos_deployment_target" not in out and "openssl_static" not in out
    assert "rust_targets=aarch64-apple-ios,aarch64-apple-ios-sim,x86_64-apple-ios" in out
    assert wh.main(base + ["packages", "--platform", "ios"]) == 0
    assert capsys.readouterr().out.split() == ["cryptography"]
    assert wh.main(["--manifest", str(root / "wheelhouse.toml"), "recipe", "--platform", "ios", "pillow"]) == 1
    assert "not built for ios" in capsys.readouterr().err


def test_workflow_and_action_wiring():
    repo = MOBILE.parents[1]
    workflow = repo / ".github" / "workflows" / "mobile-wheels.yml"
    action = repo / ".github" / "actions" / "mobile-wheelhouse" / "action.yml"
    if not workflow.is_file():
        pytest.skip("not a full repository checkout")
    text = workflow.read_text(encoding="utf-8")
    on_block = text.split("\non:", 1)[1].split("\npermissions:", 1)[0]
    assert "workflow_call:" in on_block and "workflow_dispatch:" in on_block
    assert "push:" not in on_block and "tags" not in on_block and "pull_request" not in on_block
    assert "\npermissions:\n  contents: read\n" in text.replace("\r\n", "\n")
    assert "${{ secrets." not in text and "gh release" not in text and "fury" not in text.lower()
    assert "persist-credentials: false" in text
    act = action.read_text(encoding="utf-8")
    assert "PIP_FIND_LINKS=$RUNNER_TEMP/wheelhouse" in act and "--wheelhouse \"$RUNNER_TEMP/wheelhouse\"" in act


def test_build_tag_parsing_is_pip_compatible():
    assert cmw.parse_build_tag("1000") == (1000, "")
    assert cmw.parse_build_tag("11b") == (11, "b")
    assert cmw.parse_build_tag("") == () and cmw.parse_build_tag("x1") == ()
