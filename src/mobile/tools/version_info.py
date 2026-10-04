#!/usr/bin/env python3
"""Version and build number for the mobile builds, derived from ``src/app_version.py``.

Stdlib only. ``APP_VERSION`` is read with :mod:`ast`; the module is never imported.

* ``build_version``: ``APP_VERSION``, e.g. ``9.13.6`` (Flutter ``--build-version``,
  Android versionName, iOS CFBundleShortVersionString).
* ``build_number``: ``M*1_000_000 + m*10_000 + p*100 + rebuild``, e.g. ``9130600``
  (``--build-number``, Android versionCode, iOS CFBundleVersion). It increases
  monotonically across releases. The last two digits are a rebuild counter
  (``--rebuild N`` or env ``GLOSSARION_REBUILD``, 0-99), for re-publishing the same
  version.
* ``tag``: ``v9.13.6``; ``artifact_prefix``: ``Glossarion_v9.13.6``.
* ``is_release``: ``true`` when ``GITHUB_REF`` is a ``refs/tags/v*`` tag.
* ``ref_tag``: that tag's name, or empty.
* ``tag_matches``: whether ``ref_tag`` equals ``tag``. A mismatch fails with
  ``--strict-tag``.

Usage::

    python tools/version_info.py                    # key=value lines
    python tools/version_info.py --json             # JSON object
    python tools/version_info.py --github-output    # append key=value to $GITHUB_OUTPUT (and print)
    python tools/version_info.py --get build_number # one value
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import sys
import tokenize
from pathlib import Path

MOBILE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_APP_VERSION_FILE = MOBILE_DIR.parent / "app_version.py"
ANDROID_MAX_VERSION_CODE = 2_100_000_000

_VERSION_RE = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[.\-+]?[A-Za-z0-9.\-+]*)?$")


def read_app_version(path: Path = DEFAULT_APP_VERSION_FILE) -> str:
    """Return the ``APP_VERSION = "x.y.z"`` literal from ``app_version.py`` (BOM-safe)."""
    data = Path(path).read_bytes()
    encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    text = data.decode(encoding).lstrip("﻿")
    for node in ast.parse(text, filename=str(path)).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "APP_VERSION"
                                                for t in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value.strip()
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "APP_VERSION" and isinstance(node.value, ast.Constant)):
            return str(node.value.value).strip()
    raise ValueError(f"APP_VERSION string literal not found in {path}")


def build_number(version: str, rebuild: int = 0) -> int:
    m = _VERSION_RE.match(version.strip().lstrip("vV"))
    if not m:
        raise ValueError(f"unsupported version {version!r} (expected MAJOR.MINOR.PATCH)")
    major, minor, patch = (int(x) if x is not None else 0 for x in m.groups())
    if minor > 99 or patch > 99:
        raise ValueError(f"version {version!r}: minor and patch must be <= 99 for the build-number scheme")
    if not 0 <= rebuild <= 99:
        raise ValueError(f"rebuild counter {rebuild} must be within 0..99")
    number = major * 1_000_000 + minor * 10_000 + patch * 100 + rebuild
    if number <= 0 or number > ANDROID_MAX_VERSION_CODE:
        raise ValueError(f"build number {number} is outside Android's versionCode range")
    return number


def compute(app_version_file: Path = DEFAULT_APP_VERSION_FILE, *, rebuild: int | None = None,
            github_ref: str | None = None) -> dict:
    version = read_app_version(app_version_file)
    if rebuild is None:
        rebuild = int(os.environ.get("GLOSSARION_REBUILD", "0") or 0)
    ref = os.environ.get("GITHUB_REF", "") if github_ref is None else github_ref
    ref_tag = ref[len("refs/tags/"):] if ref.startswith("refs/tags/") else ""
    tag = f"v{version}"
    return {
        "build_version": version,
        "build_number": build_number(version, rebuild),
        "tag": tag,
        "artifact_prefix": f"Glossarion_{tag}",
        "is_release": bool(ref_tag.startswith("v")),
        "ref_tag": ref_tag,
        "tag_matches": (ref_tag == tag) if ref_tag else True,
    }


def _fmt(value) -> str:
    return ("true" if value else "false") if isinstance(value, bool) else str(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Print the mobile build version/number derived from src/app_version.py.")
    ap.add_argument("--app-version-file", default=str(DEFAULT_APP_VERSION_FILE))
    ap.add_argument("--rebuild", type=int, default=None, help="rebuild counter 0-99 (default: $GLOSSARION_REBUILD or 0)")
    ap.add_argument("--json", action="store_true", help="print a JSON object")
    ap.add_argument("--github-output", action="store_true",
                    help="append key=value lines to $GITHUB_OUTPUT (printed as well)")
    ap.add_argument("--get", metavar="KEY", help="print a single value")
    ap.add_argument("--strict-tag", action="store_true", help="fail when a pushed v* tag differs from v<APP_VERSION>")
    args = ap.parse_args(argv)
    try:
        info = compute(Path(args.app_version_file), rebuild=args.rebuild)
    except (OSError, ValueError, SyntaxError) as e:
        print(f"version_info: {e}", file=sys.stderr)
        return 2
    if args.strict_tag and not info["tag_matches"]:
        print(f"version_info: tag {info['ref_tag']} does not match {info['tag']} from app_version.py", file=sys.stderr)
        return 1
    if args.get:
        if args.get not in info:
            print(f"version_info: unknown key {args.get!r} (known: {', '.join(info)})", file=sys.stderr)
            return 2
        print(_fmt(info[args.get]))
        return 0
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for k, v in info.items():
            print(f"{k}={_fmt(v)}")
    if args.github_output:
        target = os.environ.get("GITHUB_OUTPUT")
        if target:
            with open(target, "a", encoding="utf-8") as f:
                for k, v in info.items():
                    f.write(f"{k}={_fmt(v)}\n")
        else:
            print("version_info: GITHUB_OUTPUT is not set; printed only", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
