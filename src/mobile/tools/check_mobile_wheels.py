#!/usr/bin/env python3
"""Check that every mobile dependency can be installed by ``flet build`` (Android + iOS).

Stdlib only. Network access to pypi.org and pypi.flet.dev is required.

What ``flet build`` does (flet 1.0.3 + serious_python ``package``): for each target ABI it runs::

    pip install --only-binary :all: [--no-binary <source_packages>] \
        --extra-index-url https://pypi.flet.dev --target ... <[project].dependencies + [tool.flet.<os>].dependencies>

A sitecustomize fakes the platform:

* ``sysconfig.get_platform()`` returns ``android-24-arm64_v8a`` /
  ``ios-13.0-arm64-iphoneos`` / ...
* ``platform.system()`` returns ``Android`` / ``iOS``.

``sys.platform`` and ``platform.machine()`` stay those of the build host: linux/x86_64
on the Android runner, darwin/arm64 on the macOS runner. This tool mirrors that run
**for the whole dependency tree** (a transitive dependency without a wheel breaks the
build just like a direct one).

For each target, every distribution must resolve to a version satisfying all collected
specifiers, with one of:

* a binary wheel for the target: ``cp313``/``abi3``/``py3``, ``android_<=24_<abi>``
  or ``ios_<=13.0_<arch>_<sdk>``, from pypi.flet.dev or PyPI;
* a pure ``py3-none-any`` wheel;
* an sdist, only for packages listed in ``[tool.flet].source_packages``. Pip then builds
  it with ``--no-binary``, and binary wheels of that package are ignored.

Targets:

* Android: ``[tool.flet.android].target_arch`` (arm64-v8a, x86_64).
* iOS: ``[tool.flet.ios].target_arch``. When unset, serious_python installs all of
  iphoneos.arm64, iphonesimulator.arm64 and iphonesimulator.x86_64.

The report has a per-pin table showing the native wheel tags required for every
**native** direct pin: cp313 android_24_arm64_v8a, android_24_x86_64,
ios_13_0_arm64_iphoneos and ios_13_0_arm64_iphonesimulator. A **pure** direct pin must
ship a py3-none-any wheel or be a source package. It also checks that every direct pin
can be installed on the CI host (ubuntu-24.04, cp313, manylinux x86_64) from PyPI alone,
because ``uv sync`` for host tests has no Flet index. Windows/macOS host wheels are
warnings.

Usage::

    python tools/check_mobile_wheels.py [--pyproject pyproject.toml] [--report build/wheels.md]
           [--json build/wheels.json] [--direct-only] [--cache-dir build/wheel_cache]
           [--platform all|android|ios]

Self-built wheels (``ci/wheels/wheelhouse.toml``; built by .github/workflows/mobile-wheels.yml
and handed to ``flet build`` through ``PIP_FIND_LINKS``). With none of the flags below the
check behaves exactly as without a wheelhouse, so the security pins fail while no index has
their mobile wheels:

* ``--wheelhouse DIR`` (repeatable): add the wheels in DIR as a third source. Selection then
  follows pip (version, then tag priority, then build tag; an exact tie goes to the index),
  every (package, target) the manifest promises must be picked from DIR, and DIR may hold
  nothing else.
* ``--wheelhouse-plan``: the same with the promised wheels as virtual "planned" files (fails
  fast on index or dependency drift before any wheel exists).
* ``--verify-wheelhouse DIR``: offline structure check of a built wheelhouse (zip integrity,
  RECORD hashes, filename vs METADATA / WHEEL vs manifest, nothing unknown).
* ``--verify-installed SITE_PACKAGES``: offline check that every arch directory of a finished
  ``flet build`` (build/site-packages) holds the wheelhouse copy of each promised package.

Exit codes: 0 ok, 1 something is not installable, 2 usage/network error.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from email.parser import HeaderParser
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    sys.stderr.write("check_mobile_wheels.py needs Python 3.11+ (tomllib)\n")
    raise SystemExit(2)

MOBILE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PYPROJECT = MOBILE_DIR / "pyproject.toml"
DEFAULT_WHEELHOUSE_MANIFEST = MOBILE_DIR / "ci" / "wheels" / "wheelhouse.toml"
FLET_INDEX = "https://pypi.flet.dev"
PYPI_SIMPLE = "https://pypi.org/simple"
PYPI_JSON = "https://pypi.org/pypi"
TARGET_PYTHON = (3, 13)
TARGET_PYTHON_FULL = "3.13.15"  # the CPython Flet 1.0.3 bundles (python-build 20260921)
USER_AGENT = "glossarion-check-mobile-wheels/1 (+https://github.com/Shirochi-stack/Glossarion)"


# ============================================================================ PEP 440 (subset)
_VERSION_RE = re.compile(
    r"""^\s*v?(?:(?P<epoch>\d+)!)?(?P<release>\d+(?:\.\d+)*)
        (?:[-_.]?(?P<pre_l>a|alpha|b|beta|c|rc|pre|preview)[-_.]?(?P<pre_n>\d+)?)?
        (?:(?:-(?P<post_n1>\d+))|(?:[-_.]?(?P<post_l>post|rev|r)[-_.]?(?P<post_n2>\d+)?))?
        (?:[-_.]?(?P<dev_l>dev)[-_.]?(?P<dev_n>\d+)?)?
        (?:\+(?P<local>[a-z0-9]+(?:[-_.][a-z0-9]+)*))?\s*$""", re.X | re.I)
_PRE_ORDER = {"a": 0, "alpha": 0, "b": 1, "beta": 1, "c": 2, "rc": 2, "pre": 2, "preview": 2}


class InvalidVersion(ValueError):
    pass


class Version:
    __slots__ = ("epoch", "release", "pre", "post", "dev", "local", "_key", "text")

    def __init__(self, text: str):
        m = _VERSION_RE.match(text)
        if not m:
            raise InvalidVersion(text)
        self.text = text.strip()
        self.epoch = int(m.group("epoch") or 0)
        self.release = tuple(int(x) for x in m.group("release").split("."))
        self.pre = (_PRE_ORDER[m.group("pre_l").lower()], int(m.group("pre_n") or 0)) if m.group("pre_l") else None
        post = m.group("post_n1") or m.group("post_n2")
        self.post = int(post or 0) if (m.group("post_n1") or m.group("post_l")) else None
        self.dev = int(m.group("dev_n") or 0) if m.group("dev_l") else None
        self.local = tuple((int(p) if p.isdigit() else p.lower()) for p in re.split(r"[-_.]", m.group("local"))) \
            if m.group("local") else None
        rel = list(self.release)
        while len(rel) > 1 and rel[-1] == 0:
            rel.pop()
        if self.pre is None and self.post is None and self.dev is not None:
            pre_k = (-1, 0)
        elif self.pre is None:
            pre_k = (9, 0)
        else:
            pre_k = self.pre
        post_k = (-1,) if self.post is None else (self.post,)
        dev_k = (float("inf"),) if self.dev is None else (self.dev,)
        local_k = () if self.local is None else tuple((1, x) if isinstance(x, int) else (0, x) for x in self.local)
        self._key = (self.epoch, tuple(rel), pre_k, post_k, dev_k, local_k)

    @property
    def is_prerelease(self) -> bool:
        return self.pre is not None or self.dev is not None

    @property
    def public(self) -> "Version":
        return Version(self.text.split("+")[0]) if self.local else self

    @property
    def base(self) -> tuple:
        return (self.epoch, self.release)

    def __eq__(self, o):
        return isinstance(o, Version) and self._key == o._key

    def __lt__(self, o):
        return self._key < o._key

    def __le__(self, o):
        return self._key <= o._key

    def __gt__(self, o):
        return self._key > o._key

    def __ge__(self, o):
        return self._key >= o._key

    def __hash__(self):
        return hash(self._key)

    def __repr__(self):
        return f"Version({self.text!r})"

    def __str__(self):
        return self.text


def parse_version(text: str) -> Version | None:
    try:
        return Version(text)
    except InvalidVersion:
        return None


def _release_prefix_match(cand: Version, prefix: str) -> bool:
    pv = parse_version(prefix)
    if pv is None:
        return False
    n = len(pv.release)
    crel = cand.release + (0,) * max(0, n - len(cand.release))
    return cand.epoch == pv.epoch and crel[:n] == pv.release


class Specifier:
    _RE = re.compile(r"^\s*(~=|==|!=|<=|>=|<|>|===)\s*(\S+?)\s*$")

    def __init__(self, text: str):
        m = self._RE.match(text)
        if not m:
            raise ValueError(f"bad specifier {text!r}")
        self.op, self.ver = m.group(1), m.group(2)
        self.wild = self.ver.endswith(".*")
        self.v = None if self.op == "===" else parse_version(self.ver[:-2] if self.wild else self.ver)
        if self.op != "===" and self.v is None:
            raise ValueError(f"bad version in specifier {text!r}")

    @property
    def mentions_prerelease(self) -> bool:
        return bool(self.v and self.v.is_prerelease)

    def contains(self, c: Version) -> bool:
        op, v = self.op, self.v
        if op == "===":
            return str(c) == self.ver
        if op in ("==", "!="):
            if self.wild:
                r = _release_prefix_match(c.public, self.ver[:-2])
            else:
                r = (c if v.local else c.public) == v
            return r if op == "==" else not r
        if op == "~=":
            prefix = ".".join(str(x) for x in v.release[:-1])
            return c.public >= v and _release_prefix_match(c.public, prefix)
        cp = c.public
        if op == ">=":
            return cp >= v
        if op == "<=":
            return cp <= v
        if op == "<":
            if not cp < v:
                return False
            return not (cp.is_prerelease and not v.is_prerelease and cp.base == v.base)
        if op == ">":
            if not cp > v:
                return False
            if cp.post is not None and v.post is None and cp.base == v.base and cp.pre is None and v.pre is None:
                return False
            return True
        return False

    def __str__(self):
        return f"{self.op}{self.ver}"


class SpecifierSet:
    def __init__(self, text: str = ""):
        self.specs = [Specifier(p) for p in text.split(",") if p.strip()]

    def contains(self, v: Version, prereleases: bool = False) -> bool:
        if v.is_prerelease and not prereleases and not any(s.mentions_prerelease for s in self.specs):
            return False
        return all(s.contains(v) for s in self.specs)

    def __str__(self):
        return ",".join(str(s) for s in self.specs)

    def __bool__(self):
        return bool(self.specs)


# ============================================================================ PEP 508 (subset)
def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


_REQ_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$")


@dataclass
class Requirement:
    name: str
    extras: frozenset
    spec: SpecifierSet
    marker: str | None
    url: str | None
    raw: str

    @property
    def key(self) -> str:
        return canonical(self.name)


def parse_requirement(text: str) -> Requirement:
    raw = text.strip()
    body, marker = raw, None
    if ";" in raw:
        body, marker = raw.split(";", 1)
        marker = marker.strip() or None
    m = _REQ_RE.match(body)
    if not m:
        raise ValueError(f"cannot parse requirement {text!r}")
    name, extras, rest = m.group(1), m.group(2), m.group(3).strip()
    url = None
    if rest.startswith("@"):
        url, rest = rest[1:].strip(), ""
    if rest.startswith("(") and rest.endswith(")"):
        rest = rest[1:-1]
    ex = frozenset(canonical(e) for e in (extras or "[]")[1:-1].split(",") if e.strip())
    return Requirement(name, ex, SpecifierSet(rest), marker, url, raw)


_MARKER_TOKEN = re.compile(r"""\s*(?:(?P<lp>\()|(?P<rp>\))|(?P<op>===|==|!=|<=|>=|~=|<|>|not\s+in\b|in\b)|
                                (?P<bool>and\b|or\b)|(?P<str>'[^']*'|"[^"]*")|(?P<var>[A-Za-z_][A-Za-z0-9_.]*))""", re.X)
_VERSION_VARS = {"python_version", "python_full_version", "implementation_version", "platform_release"}


def evaluate_marker(marker: str | None, env: dict, extras: frozenset = frozenset()) -> bool:
    if not marker:
        return True
    toks, pos = [], 0
    while pos < len(marker):
        if marker[pos:].strip() == "":
            break
        m = _MARKER_TOKEN.match(marker, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot parse marker {marker!r}")
        kind = m.lastgroup
        val = m.group(kind)
        toks.append((kind, re.sub(r"\s+", " ", val)))
        pos = m.end()
    idx = [0]

    def peek():
        return toks[idx[0]] if idx[0] < len(toks) else (None, None)

    def take():
        t = peek()
        idx[0] += 1
        return t

    def value(tok):
        kind, val = tok
        if kind == "str":
            return ("lit", val[1:-1])
        if kind == "var":
            return ("var", val)
        raise ValueError(f"bad marker operand in {marker!r}")

    def compare(lhs, op, rhs):
        def resolve(x):
            return x[1] if x[0] == "lit" else ("__extra__" if x[1] == "extra" else env.get(x[1], ""))
        lv, rv = resolve(lhs), resolve(rhs)
        if lv == "__extra__" or rv == "__extra__":
            other = rv if lv == "__extra__" else lv
            hit = canonical(other) in extras
            return hit if op == "==" else (not hit) if op == "!=" else False
        var = lhs[1] if lhs[0] == "var" else rhs[1] if rhs[0] == "var" else ""
        if op in ("in", "not in"):
            r = lv in rv
            return r if op == "in" else not r
        if var in _VERSION_VARS and op in ("==", "!=", "<", "<=", ">", ">=", "~=", "==="):
            target = parse_version(lv) if lhs[0] == "var" else None
            try:
                if target is not None:
                    return Specifier(f"{op}{rv}").contains(target)
                lver = parse_version(lv)
                if lver is not None:
                    return Specifier(f"{op}{rv}").contains(lver)
            except ValueError:
                pass
        if op == "==" or op == "===":
            return lv == rv
        if op == "!=":
            return lv != rv
        return False

    def atom():
        kind, _ = peek()
        if kind == "lp":
            take()
            r = expr()
            if take()[0] != "rp":
                raise ValueError(f"unbalanced marker {marker!r}")
            return r
        lhs = value(take())
        op = take()
        if op[0] != "op":
            raise ValueError(f"bad marker operator in {marker!r}")
        rhs = value(take())
        return compare(lhs, op[1], rhs)

    def conj():
        r = atom()
        while peek() == ("bool", "and"):
            take()
            r = atom() and r
        return r

    def expr():
        r = conj()
        while peek() == ("bool", "or"):
            take()
            r = conj() or r
        return r

    result = expr()
    if idx[0] != len(toks):
        raise ValueError(f"trailing tokens in marker {marker!r}")
    return bool(result)


# ============================================================================ targets / tags
@dataclass(frozen=True)
class Target:
    key: str
    os: str            # android | ios | host
    platform_re: str   # regex on a single platform tag; groups: version parts
    max_version: tuple
    markers: dict
    label: str
    required: bool = True

    def platform_ok(self, plat: str) -> bool:
        if plat == "any":
            return True
        m = re.fullmatch(self.platform_re, plat)
        if not m:
            return False
        if self.os == "host-macos":
            return True
        ver = tuple(int(x) for x in m.groups() if x is not None)
        return ver <= self.max_version


def _env(platform_system: str, sys_platform: str, machine: str) -> dict:
    return {
        "python_version": f"{TARGET_PYTHON[0]}.{TARGET_PYTHON[1]}",
        "python_full_version": TARGET_PYTHON_FULL,
        "implementation_name": "cpython",
        "implementation_version": TARGET_PYTHON_FULL,
        "platform_python_implementation": "CPython",
        "os_name": "posix" if platform_system != "Windows" else "nt",
        "sys_platform": sys_platform,
        "platform_system": platform_system,
        "platform_machine": machine,
        "platform_release": "",
        "platform_version": ";embedded",
    }


ANDROID_ENV = _env("Android", "linux", "x86_64")   # flet build on ubuntu: sys.platform/machine are the host's
IOS_ENV = _env("iOS", "darwin", "arm64")            # flet build on macOS
TARGETS = {
    "arm64-v8a": Target("android-arm64", "android", r"android_(\d+)_arm64_v8a", (24,), ANDROID_ENV,
                        "android_24_arm64_v8a"),
    "x86_64": Target("android-x86_64", "android", r"android_(\d+)_x86_64", (24,), ANDROID_ENV, "android_24_x86_64"),
    "armeabi-v7a": Target("android-armv7", "android", r"android_(\d+)_armeabi_v7a", (24,), ANDROID_ENV,
                          "android_24_armeabi_v7a"),
    "iphoneos.arm64": Target("ios-device", "ios", r"ios_(\d+)_(\d+)_arm64_iphoneos", (13, 0), IOS_ENV,
                             "ios_13_0_arm64_iphoneos"),
    "iphonesimulator.arm64": Target("ios-sim-arm64", "ios", r"ios_(\d+)_(\d+)_arm64_iphonesimulator", (13, 0), IOS_ENV,
                                    "ios_13_0_arm64_iphonesimulator"),
    "iphonesimulator.x86_64": Target("ios-sim-x86_64", "ios", r"ios_(\d+)_(\d+)_x86_64_iphonesimulator", (13, 0),
                                     IOS_ENV, "ios_13_0_x86_64_iphonesimulator"),
}
IOS_DEFAULT_ARCHS = ["iphoneos.arm64", "iphonesimulator.arm64", "iphonesimulator.x86_64"]
ANDROID_DEFAULT_ARCHS = ["arm64-v8a", "x86_64"]
_MANYLINUX_LEGACY = {"manylinux1": (2, 5), "manylinux2010": (2, 12), "manylinux2014": (2, 17)}
HOST_TARGETS = [
    Target("host-linux", "host-linux", r"manylinux_(\d+)_(\d+)_x86_64", (2, 39),
           _env("Linux", "linux", "x86_64"), "cp313 manylinux x86_64 (CI host)", True),
    Target("host-windows", "host-windows", r"win_amd64", (), _env("Windows", "win32", "AMD64"),
           "cp313 win_amd64 (local dev)", False),
    Target("host-macos", "host-macos", r"macosx_(\d+)_(\d+)_(?:arm64|universal2)", (), _env("Darwin", "darwin", "arm64"),
           "cp313 macOS arm64 (local dev)", False),
]


def _py_abi_ok(py: str, abi: str) -> bool:
    maj, mnr = TARGET_PYTHON
    pys, abis = py.split("."), abi.split(".")
    for p in pys:
        for a in abis:
            if a == "none" and (p in ("py3", f"py{maj}{mnr}", f"cp{maj}{mnr}") or
                                (re.fullmatch(r"py3\d+", p) and int(p[3:]) <= mnr)):
                return True
            if a == "abi3" and re.fullmatch(r"cp3\d+", p) and 2 <= int(p[3:]) <= mnr:
                return True
            if a == f"cp{maj}{mnr}" and p == f"cp{maj}{mnr}":
                return True
    return False


@dataclass
class DistFile:
    filename: str
    version: Version
    source: str              # pypi | flet | wheelhouse | planned
    kind: str                # wheel | sdist
    py: str = ""
    abi: str = ""
    plats: tuple = ()
    requires_python: str = ""
    yanked: bool = False
    build: tuple = ()        # wheel build tag as pip compares it: (number, suffix), () when absent
    requires_dist: list | None = None   # own metadata (wheelhouse / planned); None: ask PyPI JSON
    path: str = ""           # local file (wheelhouse)

    @property
    def pure(self) -> bool:
        return self.kind == "wheel" and self.plats == ("any",)


def parse_build_tag(text: str) -> tuple:
    """A wheel build tag the way pip orders it: ``(1000, "")``; ``()`` when absent or invalid."""
    m = re.fullmatch(r"(\d+)(.*)", text or "")
    return (int(m.group(1)), m.group(2)) if m else ()


def parse_filename(filename: str, project: str, source: str, requires_python: str = "", yanked: bool = False):
    fn = filename.split("#")[0]
    if fn.endswith(".whl"):
        parts = fn[:-4].split("-")
        build = ""
        if len(parts) == 6:
            name, ver, build, py, abi, plat = parts
        elif len(parts) == 5:
            name, ver, py, abi, plat = parts
        else:
            return None
        if canonical(name) != canonical(project):
            return None
        v = parse_version(ver)
        if v is None:
            return None
        return DistFile(fn, v, source, "wheel", py, abi, tuple(plat.split(".")), requires_python, yanked,
                        parse_build_tag(build))
    m = re.match(r"^(?P<name>.+?)-(?P<ver>\d[^-]*?)\.(?:tar\.gz|zip|tar\.bz2|tgz)$", fn)
    if m and canonical(m.group("name")) == canonical(project):
        v = parse_version(m.group("ver"))
        if v is not None:
            return DistFile(fn, v, source, "sdist", requires_python=requires_python, yanked=yanked)
    return None


def requires_python_ok(spec: str) -> bool:
    if not spec:
        return True
    try:
        return SpecifierSet(spec.replace(" ", "")).contains(Version(TARGET_PYTHON_FULL), prereleases=True)
    except ValueError:
        return True


def file_ok(f: DistFile, target: Target, source_package: bool) -> bool:
    if f.yanked or not requires_python_ok(f.requires_python):
        return False
    if source_package and target.os in ("android", "ios"):
        return f.kind == "sdist"
    if f.kind != "wheel":
        return False
    if not _py_abi_ok(f.py, f.abi):
        return False
    if target.os == "host-linux":
        for plat in f.plats:
            if plat == "any":
                return True
            m = re.fullmatch(r"(manylinux1|manylinux2010|manylinux2014)_x86_64", plat)
            if m and _MANYLINUX_LEGACY[m.group(1)] <= target.max_version:
                return True
            if target.platform_ok(plat):
                return True
        return False
    return any(target.platform_ok(p) for p in f.plats)


def _tag_groups() -> tuple:
    """(interpreter, abi) groups in pip's preference order (packaging.tags.sys_tags for cp313).

    Platform-specific groups come first: cp313-cp313, cp313-abi3, cp313-none, cp312-abi3 ...
    cp32-abi3, py313-none, py3-none, py312-none ... py30-none. Then the ``any`` groups:
    cp313-none-any, py313-none-any, py3-none-any, py312-none-any ... py30-none-any.
    """
    maj, mnr = TARGET_PYTHON
    cp = f"cp{maj}{mnr}"
    py_range = [f"py{maj}{mnr}", f"py{maj}"] + [f"py{maj}{m}" for m in range(mnr - 1, -1, -1)]
    specific = [(cp, cp), (cp, "abi3"), (cp, "none")] + [(f"cp{maj}{m}", "abi3") for m in range(mnr - 1, 1, -1)]
    specific += [(p, "none") for p in py_range]
    generic = [(cp, "none")] + [(p, "none") for p in py_range]
    return tuple(specific), tuple(generic)


_TAG_GROUPS = _tag_groups()


def tag_priority(f: DistFile, target: Target) -> tuple | None:
    """How much pip prefers ``f``'s best tag for ``target`` (higher is better); None if no tag fits.

    Group order follows packaging.tags; within a group a higher platform level ranks higher
    (android_24 above android_21). Sdists rank below every wheel.
    """
    if f.kind != "wheel":
        return (-(len(_TAG_GROUPS[0]) + len(_TAG_GROUPS[1]) + 1), ())
    specific, generic = _TAG_GROUPS
    best = None
    for py in f.py.split("."):
        for abi in f.abi.split("."):
            for plat in f.plats:
                if plat == "any":
                    if (py, abi) not in generic:
                        continue
                    key = (-(len(specific) + generic.index((py, abi))), ())
                else:
                    if (py, abi) not in specific:
                        continue
                    m = re.fullmatch(target.platform_re, plat)
                    if not m or not target.platform_ok(plat):
                        continue
                    key = (-specific.index((py, abi)), tuple(int(x) for x in m.groups() if x is not None))
                if best is None or key > best:
                    best = key
    return best


def pip_rank(f: DistFile, target: Target) -> tuple:
    """pip's preference among the files of one version: tag priority, then build tag; an exact tie
    goes to the index (pip lists find-links files first and keeps the last best candidate)."""
    return (tag_priority(f, target) or (-10 ** 6, ()), f.build, f.source in ("pypi", "flet"))


# ============================================================================ fetching
class Fetcher:
    def __init__(self, cache_dir: Path | None = None, timeout: float = 30.0, ttl: float = 6 * 3600, jobs: int = 16):
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.ttl = ttl
        self.jobs = jobs
        self._mem: dict[str, object] = {}
        self._flet_projects: dict[str, str] | None = None
        self.requests = 0
        # canonical name -> [DistFile] from --wheelhouse / --wheelhouse-plan, offered next to the index files
        self.extra_files: dict[str, list] = {}

    def _get(self, url: str, accept: str | None = None) -> tuple[int, bytes]:
        key = hashlib.sha1(f"{url}|{accept}".encode()).hexdigest()
        if self.cache_dir:
            p = self.cache_dir / key
            if p.exists() and time.time() - p.stat().st_mtime < self.ttl:
                data = p.read_bytes()
                return int(data[:3]), data[4:]
        headers = {"User-Agent": USER_AGENT}
        if accept:
            headers["Accept"] = accept
        last = None
        for attempt in range(4):
            try:
                self.requests += 1
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=self.timeout) as r:
                    status, body = r.status, r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    status, body = 404, b""
                    break
                last = e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = e
            time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"GET {url} failed: {last}")
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / key).write_bytes(f"{status:03d} ".encode() + body)
        return status, body

    def flet_projects(self) -> dict:
        if self._flet_projects is None:
            status, body = self._get(f"{FLET_INDEX}/")
            if status != 200:
                raise RuntimeError(f"{FLET_INDEX}/ returned HTTP {status}")
            projects = {}
            for href, text in re.findall(r'<a\s+[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', body.decode("utf-8", "replace")):
                projects.setdefault(canonical(text.strip()), href)
            self._flet_projects = projects
        return self._flet_projects

    def files(self, project: str) -> list:
        extra = self.extra_files.get(canonical(project))
        index_files = self._index_files(project)
        return index_files + extra if extra else index_files

    def _index_files(self, project: str) -> list:
        key = f"files:{canonical(project)}"
        if key in self._mem:
            return self._mem[key]
        out: list[DistFile] = []
        status, body = self._get(f"{PYPI_SIMPLE}/{canonical(project)}/", "application/vnd.pypi.simple.v1+json")
        if status == 200:
            for f in json.loads(body).get("files", []):
                df = parse_filename(f["filename"], project, "pypi", f.get("requires-python") or "",
                                    bool(f.get("yanked")))
                if df:
                    out.append(df)
        href = self.flet_projects().get(canonical(project))
        if href:
            status, body = self._get(href if href.startswith("http") else f"{FLET_INDEX}/{href.strip('/')}/")
            if status == 200:
                for attrs, text in re.findall(r"<a\s+([^>]*)>([^<]+)</a>", body.decode("utf-8", "replace")):
                    rp = re.search(r'data-requires-python="([^"]*)"', attrs)
                    df = parse_filename(text.strip(), project, "flet", html.unescape(rp.group(1)) if rp else "",
                                        "data-yanked" in attrs)
                    if df:
                        out.append(df)
        self._mem[key] = out
        return out

    def requires_dist(self, project: str, version: Version):
        key = f"meta:{canonical(project)}:{version}"
        if key in self._mem:
            return self._mem[key]
        status, body = self._get(f"{PYPI_JSON}/{canonical(project)}/{version}/json")
        result = None
        if status == 200:
            info = json.loads(body).get("info", {})
            result = list(info.get("requires_dist") or [])
        self._mem[key] = result
        return result

    def prefetch(self, fn, items) -> None:
        items = list(items)
        if len(items) <= 1:
            for it in items:
                fn(*it) if isinstance(it, tuple) else fn(it)
            return
        with ThreadPoolExecutor(max_workers=self.jobs) as ex:
            list(ex.map(lambda it: fn(*it) if isinstance(it, tuple) else fn(it), items))


# ============================================================================ resolution
@dataclass
class Choice:
    name: str
    version: Version | None
    file: DistFile | None
    reason: str = ""
    parents: list = field(default_factory=list)


class Resolver:
    """Greedy pip-like resolution (newest compatible version; no backtracking) for one target.

    ``rank="legacy"`` reports a pypi.flet.dev file first within a version (the historic report);
    ``rank="pip"`` (wheelhouse modes) picks the file pip would install (see ``pip_rank``).
    """

    def __init__(self, target: Target, fetcher: Fetcher, source_packages: set, local_packages: set,
                 transitive: bool = True, rank: str = "legacy"):
        self.t = target
        self.f = fetcher
        self.source = source_packages
        self.local = local_packages
        self.transitive = transitive
        self.rank = rank
        self.specs: dict[str, list] = {}       # name -> [(Requirement, parent, parent_version)]
        self.chosen: dict[str, Choice] = {}
        self.walked: dict[str, tuple] = {}
        self.errors: list[str] = []

    def _active(self, name: str) -> list:
        out = []
        for req, parent, pver in self.specs.get(name, []):
            if parent is None or (parent in self.chosen and self.chosen[parent].version == pver):
                out.append((req, parent))
        return out

    def pick(self, name: str) -> Choice:
        active = self._active(name)
        files = [f for f in self.f.files(name)]
        source_pkg = name in self.source
        compat = [f for f in files if file_ok(f, self.t, source_pkg)]
        by_version: dict[Version, list] = {}
        for f in compat:
            by_version.setdefault(f.version, []).append(f)
        sets = [r.spec for r, _ in active]
        parents = sorted({p or "<pyproject>" for _, p in active})

        def satisfies(v, pre):
            return all(s.contains(v, prereleases=pre) for s in sets)

        versions = sorted(by_version, reverse=True)
        pick = next((v for v in versions if satisfies(v, False)), None)
        if pick is None:
            pick = next((v for v in versions if satisfies(v, True)), None)
        if pick is None:
            all_versions = sorted({f.version for f in files if satisfies(f.version, True)}, reverse=True)
            if not files:
                why = "not found on PyPI or pypi.flet.dev"
            elif not all_versions:
                why = f"no release satisfies {' & '.join(str(s) or '*' for s in sets)}"
            else:
                v = all_versions[0]
                tags = sorted({f.filename for f in files if f.version == v})
                hint = "only an sdist (add it to source_packages if it is pure Python)" if all(
                    f.kind == "sdist" for f in files if f.version == v) else f"no {self.t.label}/cp313 wheel"
                if source_pkg:
                    hint = "no sdist, but it is listed in source_packages"
                usable = sorted({f.version for f in compat}, reverse=True)[:5]
                on_flet = canonical(name) in self.f.flet_projects()
                why = (f"{v} has {hint} (e.g. {', '.join(tags[:3])}{' ...' if len(tags) > 3 else ''}); "
                       f"versions installable here: {', '.join(map(str, usable)) or 'none'}"
                       f"{'' if on_flet else '; not on pypi.flet.dev'}")
            return Choice(name, None, None, why, parents)
        if self.rank == "pip":
            best = max(by_version[pick], key=lambda f: pip_rank(f, self.t))
        else:
            best = sorted(by_version[pick], key=lambda f: (f.pure, f.source != "flet", f.kind != "wheel"))[0]
        return Choice(name, pick, best, "", parents)

    def _own_metadata(self, name: str) -> bool:
        c = self.chosen.get(name)
        return c is not None and c.file is not None and c.file.requires_dist is not None

    def _deps(self, name: str, version: Version):
        """Requires-Dist of the chosen file: its own metadata (wheelhouse / planned) or PyPI JSON."""
        if self._own_metadata(name):
            return list(self.chosen[name].file.requires_dist)
        return self.f.requires_dist(name, version)

    def resolve(self, roots: list) -> None:
        for req in roots:
            self.specs.setdefault(req.key, []).append((req, None, None))
        pending = list(dict.fromkeys(req.key for req in roots))
        rounds = 0
        while pending and rounds < 50:
            rounds += 1
            names = [n for n in dict.fromkeys(pending) if n not in self.local]
            for n in dict.fromkeys(pending):
                if n in self.local:
                    self.chosen[n] = Choice(n, None, None, "local (dev_packages)", [])
            pending = []
            self.f.prefetch(self.f.files, names)
            changed = []
            for n in names:
                c = self.pick(n)
                prev = self.chosen.get(n)
                self.chosen[n] = c
                extras = frozenset().union(*(r.extras for r, _ in self._active(n))) if self._active(n) else frozenset()
                state = (c.version, extras)
                if c.version is not None and self.walked.get(n) != state:
                    changed.append((n, c.version, extras))
                elif prev and prev.version != c.version:
                    changed.append((n, c.version, extras))
            if not self.transitive:
                continue
            self.f.prefetch(self.f.requires_dist, [(n, v) for n, v, _ in changed
                                                    if v is not None and not self._own_metadata(n)])
            for n, v, extras in changed:
                self.walked[n] = (v, extras)
                if v is None:
                    continue
                deps = self._deps(n, v)
                if deps is None:
                    self.chosen[n].reason = f"{v} has no PyPI metadata; dependencies not checked"
                    continue
                for d in deps:
                    try:
                        req = parse_requirement(d)
                    except ValueError:
                        continue
                    try:
                        ok = evaluate_marker(req.marker, self.t.markers, extras)
                    except ValueError:
                        ok = True
                    if not ok or req.url:
                        continue
                    self.specs.setdefault(req.key, []).append((req, n, v))
                    pending.append(req.key)
            # anything whose active spec set changed must be re-picked
            for name in list(self.chosen):
                c = self.chosen[name]
                if c.version is not None and name not in self.local:
                    if not all(r.spec.contains(c.version, prereleases=True) for r, _ in self._active(name)):
                        pending.append(name)
        if pending:
            self.errors.append(f"resolution did not converge after {rounds} rounds (still pending: "
                               f"{', '.join(sorted(set(pending))[:10])})")
        for name, c in sorted(self.chosen.items()):
            if c.version is None and name not in self.local and self._active(name):
                self.errors.append(f"{name} ({', '.join(str(r.spec) or '*' for r, _ in self._active(name))}; "
                                   f"needed by {', '.join(c.parents)}): {c.reason}")


# ============================================================================ pyproject
@dataclass
class Project:
    common: list
    android: list
    ios: list
    source_packages: set
    dev_packages: dict
    android_archs: list
    ios_archs: list


def load_project(path: Path) -> Project:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    flet = data.get("tool", {}).get("flet", {})
    android, ios = flet.get("android", {}), flet.get("ios", {})
    common = [parse_requirement(r) for r in data.get("project", {}).get("dependencies", [])]
    dev_packages = {canonical(k): v for k, v in (flet.get("dev_packages") or {}).items()}
    for name, rel in list(dev_packages.items()):  # dependencies of in-repo packages
        pp = (path.parent / rel / "pyproject.toml")
        if pp.exists():
            with open(pp, "rb") as f:
                sub = tomllib.load(f)
            for r in sub.get("project", {}).get("dependencies", []):
                req = parse_requirement(r)
                if req.key not in {c.key for c in common}:
                    common.append(req)
    src = set(canonical(x) for x in (flet.get("source_packages") or []))
    return Project(
        common=common,
        android=[parse_requirement(r) for r in android.get("dependencies", [])],
        ios=[parse_requirement(r) for r in ios.get("dependencies", [])],
        source_packages=src | set(canonical(x) for x in (android.get("source_packages") or [])),
        dev_packages=dev_packages,
        android_archs=list(android.get("target_arch") or flet.get("target_arch") or ANDROID_DEFAULT_ARCHS),
        ios_archs=list(ios.get("target_arch") or IOS_DEFAULT_ARCHS),
    )


# ============================================================================ wheelhouse (self-built wheels)
PLATFORMS = ("all", "android", "ios")
WHEELHOUSE_SOURCES = ("wheelhouse", "planned")
SOURCE_LABELS = {"planned": "planned (mobile-wheels)"}
# Besides the promised wheels a wheelhouse may hold only these (pip reads every archive in a
# find-links directory, so anything else could be installed by flet build).
WHEELHOUSE_EXTRA_FILES = ("SHA256SUMS", "provenance.json")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    pass


def platform_of(target_key: str) -> str:
    """android | ios for a TARGETS key."""
    return TARGETS[target_key].os


def on_platform(target_key: str, platform: str) -> bool:
    return platform == "all" or platform_of(target_key) == platform


def has_extra_marker(req: Requirement) -> bool:
    return bool(req.marker) and re.search(r"\bextra\b", req.marker) is not None


def _marker_key(marker: str | None) -> str:
    """Spelling-independent form of a marker (quotes and spacing)."""
    if not marker:
        return ""
    out, pos = [], 0
    while pos < len(marker) and marker[pos:].strip():
        m = _MARKER_TOKEN.match(marker, pos)
        if not m or m.end() == pos:
            return " ".join(marker.split())
        kind = m.lastgroup
        val = m.group(kind)
        out.append(f'"{val[1:-1]}"' if kind == "str" else re.sub(r"\s+", " ", val))
        pos = m.end()
    return " ".join(out)


def requirement_key(text: str) -> tuple:
    """Spelling-independent form of a requirement: ``flet-libjpeg (==3.0.90)`` == ``flet_libjpeg==3.0.90``."""
    r = parse_requirement(text)
    return (r.key, tuple(sorted(r.extras)), tuple(sorted(str(s) for s in r.spec.specs)), _marker_key(r.marker),
            r.url or "")


def runtime_requirements(requires_dist) -> set:
    """The Requires-Dist entries pip installs without extras, as requirement keys."""
    out = set()
    for text in requires_dist or []:
        req = parse_requirement(text)
        if not has_extra_marker(req):
            out.add(requirement_key(text))
    return out


@dataclass(frozen=True)
class PromisedWheel:
    name: str                # canonical project name
    version: str
    recipe: str
    sdist_sha256: str
    targets: tuple           # TARGETS keys
    requires_dist: tuple     # Requires-Dist without extra markers

    def filename(self, target_key: str, build_tag: int) -> str:
        cp = f"cp{TARGET_PYTHON[0]}{TARGET_PYTHON[1]}"
        return f"{self.name.replace('-', '_')}-{self.version}-{build_tag}-{cp}-{cp}-{TARGETS[target_key].label}.whl"

    def tag(self, target_key: str) -> str:
        cp = f"cp{TARGET_PYTHON[0]}{TARGET_PYTHON[1]}"
        return f"{cp}-{cp}-{TARGETS[target_key].label}"


@dataclass
class WheelhouseManifest:
    path: Path
    build_tag: int
    wheels: list             # [PromisedWheel]
    data: dict               # the whole TOML document (toolchain, openssl, ...)

    def wheel(self, name: str):
        return next((w for w in self.wheels if w.name == canonical(name)), None)

    def promised(self, platform: str = "all") -> list:
        """[(PromisedWheel, target key)] for the targets of ``platform``."""
        return [(w, t) for w in self.wheels for t in w.targets if on_platform(t, platform)]


def load_wheelhouse_manifest(path) -> WheelhouseManifest:
    """Read and validate ci/wheels/wheelhouse.toml (raises OSError, TOMLDecodeError or ManifestError)."""
    path = Path(path)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if data.get("schema") != 1:
        raise ManifestError(f"{path}: unsupported schema {data.get('schema')!r} (expected 1)")
    build_tag = data.get("build_tag")
    if not isinstance(build_tag, int) or isinstance(build_tag, bool) or build_tag < 1:
        raise ManifestError(f"{path}: build_tag must be a positive integer")
    wheels, seen = [], set()
    for i, entry in enumerate(data.get("wheel") or []):
        where = f"{path}: [[wheel]] #{i + 1}"
        if not isinstance(entry, dict):
            raise ManifestError(f"{where} is not a table")
        name, version = entry.get("name"), entry.get("version")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise ManifestError(f"{where}: bad name {name!r}")
        if canonical(name) in seen:
            raise ManifestError(f"{where}: {name} is listed twice")
        if not isinstance(version, str) or parse_version(version) is None:
            raise ManifestError(f"{where}: bad version {version!r}")
        sdist = entry.get("sdist_sha256")
        if not isinstance(sdist, str) or not _SHA256_RE.fullmatch(sdist):
            raise ManifestError(f"{where}: sdist_sha256 must be 64 lowercase hex digits")
        targets = entry.get("targets")
        if not isinstance(targets, list) or not targets or len(set(targets)) != len(targets) or \
                any(t not in TARGETS for t in targets):
            raise ManifestError(f"{where}: targets must be distinct keys of {sorted(TARGETS)} (got {targets!r})")
        reqs = entry.get("requires_dist", [])
        if not isinstance(reqs, list) or not all(isinstance(r, str) for r in reqs):
            raise ManifestError(f"{where}: requires_dist must be a list of strings")
        for r in reqs:
            try:
                req = parse_requirement(r)
            except ValueError as e:
                raise ManifestError(f"{where}: {e}") from e
            if has_extra_marker(req):
                raise ManifestError(f"{where}: requires_dist lists only what installs without extras ({r!r})")
        recipe = entry.get("recipe", "")
        if not isinstance(recipe, str):
            raise ManifestError(f"{where}: recipe must be a string")
        seen.add(canonical(name))
        wheels.append(PromisedWheel(canonical(name), version, recipe, sdist, tuple(targets), tuple(reqs)))
    if not wheels:
        raise ManifestError(f"{path}: no [[wheel]] entries")
    return WheelhouseManifest(path, build_tag, wheels, data)


def load_uv_lock_sdists(path: Path) -> dict:
    """(canonical name, version) -> (sdist url, sha256 hex) from uv.lock."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    out = {}
    for pkg in data.get("package", []):
        sdist = pkg.get("sdist") or {}
        digest = sdist.get("hash", "")
        if pkg.get("name") and pkg.get("version") and digest.startswith("sha256:"):
            out[(canonical(pkg["name"]), str(pkg["version"]))] = (sdist.get("url", ""), digest[len("sha256:"):])
    return out


def check_manifest_pins(manifest: WheelhouseManifest, project: Project, uv_lock: Path | None) -> list:
    """The manifest must build exactly the pinned versions from the locked sdists."""
    errors = []
    reqs = project.common + project.android + project.ios
    lock = None
    if uv_lock is None or not uv_lock.is_file():
        errors.append(f"{uv_lock}: not found, so the sdist hashes in {manifest.path.name} cannot be checked")
    else:
        try:
            lock = load_uv_lock_sdists(uv_lock)
        except (OSError, tomllib.TOMLDecodeError) as e:
            errors.append(f"{uv_lock}: unreadable ({e})")
    for w in manifest.wheels:
        pins = [r for r in reqs if r.key == w.name]
        if not pins:
            errors.append(f"{w.name}: {manifest.path.name} builds {w.version} but pyproject does not pin {w.name}")
        for r in pins:
            specs = r.spec.specs
            if len(specs) != 1 or specs[0].op != "==" or specs[0].wild or \
                    parse_version(specs[0].ver) != parse_version(w.version):
                errors.append(f"{w.name}: {manifest.path.name} builds {w.version} but pyproject has {r.raw!r}")
        if lock is not None:
            locked = lock.get((w.name, w.version), (None, None))[1]
            if locked is None:
                errors.append(f"{w.name} {w.version}: no sdist hash in {uv_lock.name}")
            elif locked != w.sdist_sha256:
                errors.append(f"{w.name} {w.version}: sdist_sha256 {w.sdist_sha256} differs from {uv_lock.name} "
                              f"({locked})")
    return errors


@dataclass
class WheelhouseWheel:
    path: Path
    dist: DistFile | None = None      # the parsed filename (source "wheelhouse"), None if not a wheel name
    project: str = ""                 # canonical name from the filename
    build_text: str = ""              # build tag as written in the filename
    meta_name: str = ""
    meta_version: str = ""
    requires_dist: list = field(default_factory=list)
    wheel_build: str | None = None    # WHEEL "Build:"
    wheel_tags: list = field(default_factory=list)   # WHEEL "Tag:" lines
    dist_info: str = ""
    problems: list = field(default_factory=list)

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def tags(self) -> set:
        """The tags the filename declares (compressed tag sets expanded)."""
        if self.dist is None:
            return set()
        return {f"{py}-{abi}-{plat}" for py in self.dist.py.split(".") for abi in self.dist.abi.split(".")
                for plat in self.dist.plats}


def read_wheel(path: Path) -> WheelhouseWheel:
    """Parse a wheel's filename, METADATA and WHEEL (problems are collected, never raised)."""
    ww = WheelhouseWheel(Path(path))
    parts = ww.filename[:-4].split("-") if ww.filename.endswith(".whl") else []
    if len(parts) not in (5, 6):
        ww.problems.append(f"{ww.filename}: not a wheel filename")
        return ww
    ww.project = canonical(parts[0])
    ww.build_text = parts[2] if len(parts) == 6 else ""
    ww.dist = parse_filename(ww.filename, parts[0], "wheelhouse")
    if ww.dist is None:
        ww.problems.append(f"{ww.filename}: cannot parse the wheel filename")
        return ww
    if ww.build_text and not ww.dist.build:
        ww.problems.append(f"{ww.filename}: build tag {ww.build_text!r} does not start with a digit")
    ww.dist.path = str(ww.path)
    try:
        with zipfile.ZipFile(ww.path) as zf:
            tops = {n.split("/", 1)[0] for n in zf.namelist() if "/" in n}
            dist_infos = sorted(t for t in tops if t.endswith(".dist-info"))
            if len(dist_infos) != 1:
                ww.problems.append(f"{ww.filename}: expected one .dist-info directory, found {dist_infos}")
                return ww
            ww.dist_info = dist_infos[0]
            meta = HeaderParser().parsestr(zf.read(f"{ww.dist_info}/METADATA").decode("utf-8"))
            wheel = HeaderParser().parsestr(zf.read(f"{ww.dist_info}/WHEEL").decode("utf-8"))
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError, OSError) as e:
        ww.problems.append(f"{ww.filename}: unreadable wheel ({type(e).__name__}: {e})")
        return ww
    ww.meta_name = (meta.get("Name") or "").strip()
    ww.meta_version = (meta.get("Version") or "").strip()
    ww.requires_dist = [r.strip() for r in meta.get_all("Requires-Dist") or []]
    ww.dist.requires_dist = list(ww.requires_dist)
    ww.dist.requires_python = (meta.get("Requires-Python") or "").strip()
    build = wheel.get("Build")
    ww.wheel_build = build.strip() if build is not None else None
    ww.wheel_tags = [t.strip() for t in wheel.get_all("Tag") or []]
    v = ww.dist.version
    if canonical(ww.meta_name) != ww.project or parse_version(ww.meta_version) != v:
        ww.problems.append(f"{ww.filename}: METADATA says {ww.meta_name} {ww.meta_version}")
    di_name, _, di_version = ww.dist_info[:-len(".dist-info")].rpartition("-")
    if canonical(di_name) != ww.project or parse_version(di_version) != v:
        ww.problems.append(f"{ww.filename}: dist-info directory {ww.dist_info} does not match the filename")
    if (ww.wheel_build or "") != ww.build_text:
        ww.problems.append(f"{ww.filename}: WHEEL Build {ww.wheel_build!r} does not match the filename build tag "
                           f"{ww.build_text!r}")
    if set(ww.wheel_tags) != ww.tags:
        ww.problems.append(f"{ww.filename}: WHEEL Tag {sorted(ww.wheel_tags)} does not match the filename tags "
                           f"{sorted(ww.tags)}")
    for text in ww.requires_dist:
        try:
            parse_requirement(text)
        except ValueError:
            ww.problems.append(f"{ww.filename}: unparsable Requires-Dist {text!r}")
    return ww


def scan_wheelhouse(directory) -> tuple:
    """(wheels, problems) for a wheelhouse directory; anything but wheels and WHEELHOUSE_EXTRA_FILES is a problem."""
    directory = Path(directory)
    wheels, problems = [], []
    if not directory.is_dir():
        return wheels, [f"{directory}: not a directory"]
    for entry in sorted(directory.iterdir()):
        if entry.is_dir():
            problems.append(f"{directory}: unexpected directory {entry.name}/")
        elif entry.name.endswith(".whl"):
            wheels.append(read_wheel(entry))
        elif entry.name not in WHEELHOUSE_EXTRA_FILES:
            problems.append(f"{directory}: unexpected file {entry.name} (pip reads every archive in a "
                            f"find-links directory)")
    return wheels, problems


def _wheel_target(ww: WheelhouseWheel, promised: PromisedWheel) -> str | None:
    """The promised target whose tag is exactly the wheel's single tag, if any."""
    if len(ww.tags) != 1:
        return None
    tag = next(iter(ww.tags))
    return next((t for t in promised.targets if promised.tag(t) == tag), None)


def check_wheels_against_manifest(wheels: list, manifest: WheelhouseManifest, platform: str = "all") -> list:
    """Every promised (package, target) of ``platform`` exactly once, nothing unknown, metadata as promised."""
    errors, found = [], {}
    for ww in wheels:
        errors.extend(ww.problems)
        if ww.dist is None:
            continue
        w = manifest.wheel(ww.project)
        if w is None:
            errors.append(f"unknown wheel {ww.filename}: {manifest.path.name} does not promise {ww.project}")
            continue
        if parse_version(w.version) != ww.dist.version:
            errors.append(f"unknown wheel {ww.filename}: {manifest.path.name} promises {w.name} {w.version}")
            continue
        target = _wheel_target(ww, w)
        if target is None:
            errors.append(f"unknown wheel {ww.filename}: its Tag is not {w.name}'s promised "
                          f"cp313-cp313-<target> ({', '.join(TARGETS[t].label for t in w.targets)})")
            continue
        if ww.build_text != str(manifest.build_tag):
            errors.append(f"{ww.filename}: build tag {ww.build_text or '(none)'} is not build_tag "
                          f"{manifest.build_tag}")
        if ww.wheel_build is not None and ww.wheel_build != str(manifest.build_tag):
            errors.append(f"{ww.filename}: WHEEL Build {ww.wheel_build} is not build_tag {manifest.build_tag}")
        try:
            got = runtime_requirements(ww.requires_dist)
        except ValueError:
            got = None
        want = runtime_requirements(w.requires_dist)
        if got is not None and got != want:
            errors.append(f"{ww.filename}: Requires-Dist without extras is "
                          f"{sorted(r for r in ww.requires_dist if not has_extra_marker(parse_requirement(r)))}, "
                          f"{manifest.path.name} promises {list(w.requires_dist)}")
        found.setdefault((w.name, target), []).append(ww.filename)
    for (name, target), files in sorted(found.items()):
        if len(files) > 1:
            errors.append(f"{name} for {target}: {len(files)} wheels ({', '.join(files)})")
    for w, target in manifest.promised(platform):
        if (w.name, target) not in found:
            errors.append(f"missing {w.filename(target, manifest.build_tag)} ({w.name} {w.version} for {target})")
    return errors


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sha256sums(directory: Path, wheels: list, required: bool = False) -> list:
    """SHA256SUMS (sha256sum format) must list exactly the wheels, with their hashes."""
    sums = Path(directory) / "SHA256SUMS"
    if not sums.is_file():
        return [f"{sums}: missing"] if required else []
    errors, listed = [], {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = re.fullmatch(r"([0-9a-f]{64}) [ *]([^/\\]+)", line.strip())
        if not m:
            errors.append(f"{sums.name}: bad line {line!r}")
            continue
        listed[m.group(2)] = m.group(1)
    names = {ww.filename for ww in wheels}
    for name in sorted(set(listed) - names):
        errors.append(f"{sums.name} lists {name}, which is not in the wheelhouse")
    for ww in wheels:
        if ww.filename not in listed:
            errors.append(f"{sums.name} does not list {ww.filename}")
        elif file_sha256(ww.path) != listed[ww.filename]:
            errors.append(f"{ww.filename}: sha256 differs from {sums.name}")
    return errors


def unsafe_member_name(name: str) -> bool:
    parts = name.replace("\\", "/").split("/")
    return name.startswith(("/", "\\")) or ".." in parts or (len(name) > 1 and name[1] == ":")


def check_wheel_record(ww: WheelhouseWheel) -> list:
    """Zip CRCs, safe member names and every member's sha256 / size against RECORD."""
    errors = []
    try:
        with zipfile.ZipFile(ww.path) as zf:
            bad = zf.testzip()
            if bad is not None:
                return [f"{ww.filename}: CRC error in {bad}"]
            names = [n for n in zf.namelist() if not n.endswith("/")]
            unsafe = [n for n in names if unsafe_member_name(n)]
            if unsafe:
                errors.append(f"{ww.filename}: unsafe member names {unsafe[:5]}")
            if not ww.dist_info:
                return errors
            record_name = f"{ww.dist_info}/RECORD"
            if record_name not in names:
                return errors + [f"{ww.filename}: no {record_name}"]
            listed = {}
            for row in csv.reader(io.StringIO(zf.read(record_name).decode("utf-8"))):
                if not row:
                    continue
                if len(row) != 3:
                    errors.append(f"{ww.filename}: malformed RECORD row {row!r}")
                    continue
                listed[row[0]] = (row[1], row[2])
            unsigned = {record_name, f"{ww.dist_info}/RECORD.jws", f"{ww.dist_info}/RECORD.p7s"}
            for name in names:
                if name in unsigned:
                    continue
                if name not in listed:
                    errors.append(f"{ww.filename}: {name} is not in RECORD")
                    continue
                digest, size = listed[name]
                algo, _, value = digest.partition("=")
                data = zf.read(name)
                if algo != "sha256":
                    errors.append(f"{ww.filename}: RECORD hash of {name} is not sha256 ({digest!r})")
                elif base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode() != value:
                    errors.append(f"{ww.filename}: {name} does not match its RECORD hash")
                if size and size != str(len(data)):
                    errors.append(f"{ww.filename}: {name} is {len(data)} bytes, RECORD says {size}")
            missing = sorted(set(listed) - set(names) - unsigned)
            if missing:
                errors.append(f"{ww.filename}: RECORD lists missing members {missing[:5]}")
    except (zipfile.BadZipFile, OSError, UnicodeDecodeError, csv.Error) as e:
        errors.append(f"{ww.filename}: unreadable wheel ({type(e).__name__}: {e})")
    return errors


def verify_wheelhouse(directory, manifest: WheelhouseManifest, project: Project | None, uv_lock: Path | None,
                      platform: str = "all") -> list:
    """--verify-wheelhouse: everything a built wheelhouse must satisfy, offline."""
    wheels, problems = scan_wheelhouse(directory)
    errors = list(problems) + check_wheels_against_manifest(wheels, manifest, platform)
    for ww in wheels:
        if ww.dist is not None:
            errors.extend(check_wheel_record(ww))
    errors.extend(check_sha256sums(Path(directory), wheels))
    if project is not None:
        errors.extend(check_manifest_pins(manifest, project, uv_lock))
    return errors


def _installed_dist_info(arch_dir: Path, name: str) -> list:
    out = []
    for d in arch_dir.glob("*.dist-info"):
        dname, _, dver = d.name[:-len(".dist-info")].rpartition("-")
        if d.is_dir() and canonical(dname) == name:
            out.append((d, dver))
    return out


def verify_installed(site_packages, manifest: WheelhouseManifest, platform: str = "all",
                     wheels: list | None = None) -> tuple:
    """--verify-installed: (errors, notes). Each arch directory of a flet build's site-packages
    (SERIOUS_PYTHON_SITE_PACKAGES, build/site-packages/<arch>) must hold the wheelhouse copy of
    every promised package: its WHEEL keeps the build tag and the tag (pip copies WHEEL as is)."""
    site_packages = Path(site_packages)
    errors, notes, seen_arch = [], [], False
    by_target = {}
    for ww in wheels or []:
        w = manifest.wheel(ww.project) if ww.dist is not None else None
        t = _wheel_target(ww, w) if w is not None else None
        if t is not None:
            by_target[(w.name, t)] = ww
    for w, target in manifest.promised(platform):
        arch_dir = site_packages / target
        if not arch_dir.is_dir():
            notes.append(f"{target}: no {arch_dir} (not built for this arch)")
            continue
        seen_arch = True
        found = _installed_dist_info(arch_dir, w.name)
        if not found:
            errors.append(f"{target}: {w.name} is not installed in {arch_dir}")
            continue
        if len(found) > 1:
            errors.append(f"{target}: {len(found)} {w.name} dist-info directories in {arch_dir}")
            continue
        dist_info, version = found[0]
        expected_build, expected_tags = str(manifest.build_tag), {w.tag(target)}
        ref = by_target.get((w.name, target))
        if ref is not None:
            expected_build, expected_tags = ref.wheel_build or ref.build_text, set(ref.wheel_tags) or ref.tags
        try:
            wheel = HeaderParser().parsestr((dist_info / "WHEEL").read_text(encoding="utf-8"))
        except OSError as e:
            errors.append(f"{target}: cannot read {dist_info.name}/WHEEL ({e})")
            continue
        build = (wheel.get("Build") or "").strip()
        tags = {t.strip() for t in wheel.get_all("Tag") or []}
        if parse_version(version) != parse_version(w.version):
            errors.append(f"{target}: {w.name} {version} is installed, the wheelhouse has {w.version}")
        elif build != expected_build or tags != expected_tags:
            errors.append(f"{target}: {w.name} {version} was installed from an index, not the wheelhouse "
                          f"(WHEEL Build {build or '(none)'}, Tag {', '.join(sorted(tags)) or '(none)'}; expected "
                          f"Build {expected_build}, Tag {', '.join(sorted(expected_tags))})")
        else:
            notes.append(f"{target}: {w.name} {version} build {build} {', '.join(sorted(tags))} (wheelhouse)")
    if not seen_arch:
        errors.append(f"{site_packages}: no arch directory of {platform} targets "
                      f"({', '.join(sorted({t for _, t in manifest.promised(platform)}))})")
    return errors, notes


@dataclass
class Wheelhouse:
    """The self-built wheels taking part in a resolution (--wheelhouse or --wheelhouse-plan)."""
    manifest: WheelhouseManifest
    mode: str                                          # files | plan
    wheels: list = field(default_factory=list)         # [WheelhouseWheel] (files)
    problems: list = field(default_factory=list)       # directory problems (files)
    uv_lock: Path | None = None
    dirs: list = field(default_factory=list)           # the --wheelhouse directories (files)

    @classmethod
    def from_dirs(cls, manifest: WheelhouseManifest, dirs: list, uv_lock: Path | None) -> "Wheelhouse":
        wh = cls(manifest, "files", uv_lock=uv_lock, dirs=[Path(d) for d in dirs])
        for d in wh.dirs:
            wheels, problems = scan_wheelhouse(d)
            wh.wheels += wheels
            wh.problems += problems
        return wh

    def dist_files(self) -> dict:
        """canonical name -> [DistFile] offered to the resolver next to the index files."""
        out: dict[str, list] = {}
        if self.mode == "plan":
            for w in self.manifest.wheels:
                for t in w.targets:
                    df = parse_filename(w.filename(t, self.manifest.build_tag), w.name, "planned")
                    df.requires_dist = list(w.requires_dist)
                    out.setdefault(w.name, []).append(df)
            return out
        for ww in self.wheels:
            if ww.dist is not None and not ww.problems and self.manifest.wheel(ww.project) is not None:
                out.setdefault(ww.project, []).append(ww.dist)
        return out


def _wheelhouse_checks(project: Project, res: "CheckResult", wh: Wheelhouse, archs: list, platform: str) -> None:
    errors = check_manifest_pins(wh.manifest, project, wh.uv_lock)
    if wh.mode == "files":
        errors += wh.problems + check_wheels_against_manifest(wh.wheels, wh.manifest, platform)
        for d in wh.dirs:
            errors += check_sha256sums(d, [ww for ww in wh.wheels if ww.path.parent == d])
    for w, arch in wh.manifest.promised(platform):
        if arch not in archs:
            continue
        target = TARGETS[arch]
        expected = w.filename(arch, wh.manifest.build_tag)
        c = res.targets.get(target.key, {}).get(w.name)
        if c is None:
            status = f"not in the dependency tree of {target.key}"
            errors.append(f"[{target.key}] {w.name}: {status}")
        elif c.version is None:
            status = "not installable"
            errors.append(f"[{target.key}] {w.name}: nothing installable although the wheelhouse should provide "
                          f"{expected}")
        elif c.version != parse_version(w.version):
            status = f"pip picks {c.version}"
            errors.append(f"[{target.key}] {w.name}: pip picks {c.version}, {wh.manifest.path.name} promises "
                          f"{w.version}")
        elif c.file.source not in WHEELHOUSE_SOURCES:
            status = f"index wins: {c.file.filename}"
            errors.append(f"[{target.key}] {c.file.filename} ({c.file.source}) outranks or ties wheelhouse {expected} "
                          f"(raise build_tag)")
        else:
            status = "ok"
        res.wheelhouse.append({"target": target.key, "name": w.name, "expected": expected, "status": status,
                               "picked": c.file.filename if c is not None and c.file is not None else None,
                               "source": c.file.source if c is not None and c.file is not None else None})
    res.errors.extend(f"[wheelhouse] {e}" for e in errors)


# ============================================================================ main check
@dataclass
class CheckResult:
    targets: dict = field(default_factory=dict)       # target key -> {name: Choice}
    target_errors: dict = field(default_factory=dict)
    direct: list = field(default_factory=list)
    host: dict = field(default_factory=dict)          # name -> {target key: (version, filename) | None}
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    native: dict = field(default_factory=dict)        # direct name -> True/False
    wheelhouse: list = field(default_factory=list)    # promised (package, target) rows (wheelhouse modes)


def run_check(project: Project, fetcher: Fetcher, transitive: bool = True, host: bool = True, *,
              platform: str = "all", wheelhouse: Wheelhouse | None = None) -> CheckResult:
    """``platform`` limits the device targets (all | android | ios). ``wheelhouse`` adds the
    self-built wheels, ranks files the way pip does and checks the manifest's promises."""
    res = CheckResult()
    local = set(project.dev_packages)
    res.direct = [r for r in project.common + project.android + project.ios]
    android_archs = project.android_archs if platform in ("all", "android") else []
    ios_archs = project.ios_archs if platform in ("all", "ios") else []
    plan = [(TARGETS[a], project.common + project.android) for a in android_archs if a in TARGETS] + \
           [(TARGETS[a], project.common + project.ios) for a in ios_archs if a in TARGETS]
    for a in android_archs + ios_archs:
        if a not in TARGETS:
            res.errors.append(f"unknown target_arch {a!r}")
    rank = "legacy"
    if wheelhouse is not None:
        fetcher.extra_files = wheelhouse.dist_files()
        rank = "pip"
    fetcher.flet_projects()
    for target, roots in plan:
        roots = [r for r in roots if evaluate_marker(r.marker, target.markers)]
        rv = Resolver(target, fetcher, project.source_packages, local, transitive, rank=rank)
        rv.resolve(roots)
        res.targets[target.key] = rv.chosen
        res.target_errors[target.key] = rv.errors
        res.errors.extend(f"[{target.key}] {e}" for e in rv.errors)
    # native vs pure classification of direct pins (first target that resolved it)
    for r in res.direct:
        c = None
        for chosen in res.targets.values():
            if r.key in chosen and chosen[r.key].file is not None:
                c = chosen[r.key]
                break
        if r.key in local:
            continue
        res.native[r.key] = bool(c and not c.file.pure and c.file.kind == "wheel")
        if c and r.key not in project.source_packages and c.file.kind == "sdist":
            res.errors.append(f"{r.key}: resolves to an sdist without being a source package")
    for tkey, chosen in res.targets.items():
        for name, c in sorted(chosen.items()):
            if c.version is not None and c.reason:
                res.warnings.append(f"[{tkey}] {name}: {c.reason}")
    # host installability (uv sync on the CI host has no Flet index)
    if host:
        fetcher.prefetch(fetcher.files, [r.key for r in res.direct if r.key not in local])
        for r in res.direct:
            if r.key in local:
                continue
            res.host[r.key] = {}
            pypi_files = [f for f in fetcher.files(r.key) if f.source == "pypi"]
            for ht in HOST_TARGETS:
                if not evaluate_marker(r.marker, ht.markers):
                    res.host[r.key][ht.key] = ("n/a (marker)", "")
                    continue
                cand = [f for f in pypi_files if r.spec.contains(f.version) and not f.yanked
                        and requires_python_ok(f.requires_python)]
                newest = max((f.version for f in cand), default=None)
                ok = [f for f in cand if f.version == newest and file_ok(f, ht, False)] if newest else []
                sd = [f for f in cand if f.version == newest and f.kind == "sdist"] if newest else []
                if ok:
                    res.host[r.key][ht.key] = (str(newest), ok[0].filename)
                elif sd:
                    res.host[r.key][ht.key] = (str(newest), f"sdist only: {sd[0].filename}")
                    if ht.required:  # uv builds pure sdists fine; worth knowing, not fatal
                        res.warnings.append(f"{r.key} {newest}: no {ht.label} wheel on PyPI; uv builds it from "
                                            f"{sd[0].filename}")
                else:
                    res.host[r.key][ht.key] = None
                    msg = f"{r.key} ({r.spec or '*'}): no installable {ht.label} distribution on PyPI"
                    (res.errors if ht.required else res.warnings).append(msg)
            # device vs host version drift for range pins
            hv = res.host[r.key].get("host-linux")
            for tkey, chosen in res.targets.items():
                c = chosen.get(r.key)
                if c and c.version is not None and hv and hv[1] and str(c.version) != hv[0] and \
                        not hv[0].startswith("n/a"):
                    res.warnings.append(f"{r.key}: device ({tkey}) resolves {c.version} but the host resolves "
                                        f"{hv[0]}; pin it so host tests run the shipped version")
                    break
    if wheelhouse is not None:
        _wheelhouse_checks(project, res, wheelhouse, [a for a in android_archs + ios_archs if a in TARGETS], platform)
    return res


def render_markdown(project: Project, res: CheckResult) -> str:
    o = ["# Mobile wheel check", ""]
    tkeys = list(res.targets)
    o.append(f"- Targets: {', '.join(tkeys)} (Python {TARGET_PYTHON[0]}.{TARGET_PYTHON[1]}); "
             f"source packages: {', '.join(sorted(project.source_packages)) or '-'}")
    o.append(f"- Result: **{'FAIL' if res.errors else 'PASS'}** with {len(res.errors)} errors and "
             f"{len(res.warnings)} warnings")
    o.append("")
    if res.errors or res.warnings:
        o += ["## Problems", ""] + [f"- **error** {e}" for e in res.errors] + [f"- warning: {w}" for w in res.warnings]
        o.append("")
    o += ["## Direct pins", "", "| pin | kind | " + " | ".join(tkeys) + " | host (linux) |",
          "|---|---|" + "---|" * len(tkeys) + "---|"]
    for r in res.direct:
        if r.key in project.dev_packages:
            o.append(f"| `{r.raw}` | local | " + " | ".join("dev_packages" for _ in tkeys) + " | - |")
            continue
        kind = "source" if r.key in project.source_packages else "native" if res.native.get(r.key) else "pure"
        cells = []
        for t in tkeys:
            c = res.targets[t].get(r.key)
            if c is None:
                cells.append("n/a")
            elif c.version is None:
                cells.append("**MISSING**")
            else:
                f = c.file
                tag = "sdist" if f.kind == "sdist" else "py3-none-any" if f.pure else f"{f.py}-{'.'.join(f.plats)}"
                cells.append(f"{c.version} {tag} ({SOURCE_LABELS.get(f.source, f.source)})")
        h = res.host.get(r.key, {}).get("host-linux")
        host = "-" if h is None and r.key not in res.host else ("**MISSING**" if h is None else f"{h[0]}")
        o.append(f"| `{r.raw}` | {kind} | " + " | ".join(cells) + f" | {host} |")
    o.append("")
    names = sorted(set().union(*(set(v) for v in res.targets.values())) - {r.key for r in res.direct})
    if names:
        o += ["## Transitive dependencies", "", "| distribution | " + " | ".join(tkeys) + " | needed by |",
              "|---|" + "---|" * len(tkeys) + "---|"]
        for n in names:
            cells, parents = [], set()
            for t in tkeys:
                c = res.targets[t].get(n)
                if c is None:
                    cells.append("-")
                    continue
                parents.update(c.parents)
                if c.version is None:
                    cells.append("**MISSING**" if c.reason != "local (dev_packages)" else "local")
                else:
                    f = c.file
                    cells.append(f"{c.version} {'pure' if f.pure else f.kind if f.kind == 'sdist' else 'native'}"
                                 f"{f' ({f.source})' if f.source in ('flet',) + WHEELHOUSE_SOURCES else ''}")
            o.append(f"| `{n}` | " + " | ".join(cells) + f" | {', '.join(sorted(parents))} |")
        o.append("")
    if res.wheelhouse:
        o += ["## Self-built wheels (ci/wheels/wheelhouse.toml)", "",
              "| target | package | expected | pip picks | status |", "|---|---|---|---|---|"]
        for row in res.wheelhouse:
            picked = f"`{row['picked']}` ({SOURCE_LABELS.get(row['source'], row['source'])})" if row["picked"] else "-"
            o.append(f"| {row['target']} | {row['name']} | `{row['expected']}` | {picked} | "
                     f"{row['status'] if row['status'] == 'ok' else '**' + row['status'] + '**'} |")
        o.append("")
    return "\n".join(o) + "\n"


def result_json(res: CheckResult) -> dict:
    out = {
        "errors": res.errors,
        "warnings": res.warnings,
        "targets": {t: {n: {"version": str(c.version) if c.version else None,
                            "file": c.file.filename if c.file else None,
                            "source": c.file.source if c.file else None,
                            "reason": c.reason, "parents": c.parents} for n, c in sorted(ch.items())}
                    for t, ch in res.targets.items()},
        "host": res.host,
        "native": res.native,
    }
    if res.wheelhouse:
        out["wheelhouse"] = res.wheelhouse
    return out


def _write_report(path: str | None, text: str) -> None:
    """Write a Markdown report and append it to $GITHUB_STEP_SUMMARY."""
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text)


def _verify_main(args) -> int:
    """--verify-wheelhouse / --verify-installed (offline)."""
    try:
        manifest = load_wheelhouse_manifest(args.wheelhouse_manifest)
    except (OSError, tomllib.TOMLDecodeError, ManifestError) as e:
        print(f"check_mobile_wheels: cannot read the wheelhouse manifest: {e}", file=sys.stderr)
        return 2
    notes = []
    if args.verify_wheelhouse:
        title = f"Wheelhouse check: {args.verify_wheelhouse} ({args.platform})"
        try:
            project = load_project(Path(args.pyproject))
        except (OSError, tomllib.TOMLDecodeError, ValueError) as e:
            print(f"check_mobile_wheels: cannot read {args.pyproject}: {e}", file=sys.stderr)
            return 2
        errors = verify_wheelhouse(args.verify_wheelhouse, manifest, project, Path(args.pyproject).parent / "uv.lock",
                                   args.platform)
        wheels, _ = scan_wheelhouse(args.verify_wheelhouse)
        notes = [f"{ww.filename} sha256 {file_sha256(ww.path)}" for ww in wheels]
    else:
        title = f"Installed-wheel check: {args.verify_installed} ({args.platform})"
        reference = []
        for d in args.wheelhouse or []:
            wheels, _ = scan_wheelhouse(d)
            reference += wheels
        errors, notes = verify_installed(args.verify_installed, manifest, args.platform, reference)
    for n in notes:
        print(f"ok      {n}")
    for e in errors:
        print(f"ERROR   {e}")
    report = [f"# {title}", "", f"- Result: **{'FAIL' if errors else 'PASS'}** with {len(errors)} errors", ""]
    report += [f"- **error** {e}" for e in errors] + [f"- {n}" for n in notes]
    _write_report(args.report, "\n".join(report) + "\n")
    print(f"check_mobile_wheels: {title}: {len(errors)} errors")
    return 1 if errors else 0


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check that every mobile dependency has Android/iOS wheels "
                                             "(see the module docstring).")
    ap.add_argument("--pyproject", default=str(DEFAULT_PYPROJECT))
    ap.add_argument("--report", help="write a Markdown report (also appended to $GITHUB_STEP_SUMMARY)")
    ap.add_argument("--json", dest="json_out", help="write the JSON result")
    ap.add_argument("--direct-only", action="store_true", help="skip transitive dependencies")
    ap.add_argument("--no-host", action="store_true", help="skip the PyPI host-wheel check")
    ap.add_argument("--cache-dir", help="cache HTTP responses here for 6 hours (e.g. build/wheel_cache)")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--platform", choices=PLATFORMS, default="all",
                    help="check only the Android or only the iOS targets (default: all)")
    wh = ap.add_argument_group("self-built wheels (ci/wheels/wheelhouse.toml, .github/workflows/mobile-wheels.yml)")
    wh.add_argument("--wheelhouse", action="append", metavar="DIR",
                    help="add the wheels in DIR (repeatable); pip's ranking then decides and every promised wheel "
                         "must win")
    wh.add_argument("--wheelhouse-manifest", default=str(DEFAULT_WHEELHOUSE_MANIFEST), metavar="PATH",
                    help="the wheel manifest (default: ci/wheels/wheelhouse.toml)")
    wh.add_argument("--wheelhouse-plan", action="store_true",
                    help="like --wheelhouse, with the promised wheels as planned files (before they exist)")
    wh.add_argument("--verify-wheelhouse", metavar="DIR",
                    help="offline structure check of a built wheelhouse, then exit")
    wh.add_argument("--verify-installed", metavar="SITE_PACKAGES",
                    help="offline check that a flet build installed the wheelhouse copies (build/site-packages), "
                         "then exit; --wheelhouse DIR adds the built wheels as the reference")
    args = ap.parse_args(argv)

    if args.wheelhouse_plan and args.wheelhouse:
        ap.error("--wheelhouse and --wheelhouse-plan are mutually exclusive")
    if args.verify_wheelhouse and (args.verify_installed or args.wheelhouse or args.wheelhouse_plan):
        ap.error("--verify-wheelhouse takes no --verify-installed, --wheelhouse or --wheelhouse-plan")
    if args.verify_installed and args.wheelhouse_plan:
        ap.error("--verify-installed takes --wheelhouse DIR, not --wheelhouse-plan")
    if args.verify_wheelhouse or args.verify_installed:
        return _verify_main(args)

    try:
        project = load_project(Path(args.pyproject))
    except (OSError, tomllib.TOMLDecodeError, ValueError) as e:
        print(f"check_mobile_wheels: cannot read {args.pyproject}: {e}", file=sys.stderr)
        return 2
    wheelhouse = None
    if args.wheelhouse or args.wheelhouse_plan:
        try:
            manifest = load_wheelhouse_manifest(args.wheelhouse_manifest)
        except (OSError, tomllib.TOMLDecodeError, ManifestError) as e:
            print(f"check_mobile_wheels: cannot read the wheelhouse manifest: {e}", file=sys.stderr)
            return 2
        uv_lock = Path(args.pyproject).parent / "uv.lock"
        if args.wheelhouse_plan:
            wheelhouse = Wheelhouse(manifest, "plan", uv_lock=uv_lock)
        else:
            wheelhouse = Wheelhouse.from_dirs(manifest, args.wheelhouse, uv_lock)
    fetcher = Fetcher(Path(args.cache_dir) if args.cache_dir else None, args.timeout, jobs=args.jobs)
    started = time.time()
    try:
        res = run_check(project, fetcher, transitive=not args.direct_only, host=not args.no_host,
                        platform=args.platform, wheelhouse=wheelhouse)
    except RuntimeError as e:
        print(f"check_mobile_wheels: network error: {e}", file=sys.stderr)
        return 2
    report = render_markdown(project, res)
    _write_report(args.report, report)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result_json(res), indent=1), encoding="utf-8")
    if not args.quiet:
        for e in res.errors:
            print(f"ERROR   {e}")
        for w in res.warnings:
            print(f"WARNING {w}")
    n_dists = len(set().union(*(set(v) for v in res.targets.values()))) if res.targets else 0
    print(f"check_mobile_wheels: {len(res.direct)} direct pins, {n_dists} distributions across "
          f"{len(res.targets)} targets; {len(res.errors)} errors, {len(res.warnings)} warnings "
          f"({fetcher.requests} requests, {time.time() - started:.0f}s)")
    return 1 if res.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
