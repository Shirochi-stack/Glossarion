#!/usr/bin/env python3
"""Mobile release assets: completeness, checksums and the never-clobber guard.

Used only by the gated ``release`` job of ``.github/workflows/build-mobile.yml`` (a manual run on
a tag with ``publish_release`` ticked and the repository variable ``MOBILE_RELEASE_ENABLED=true``)::

    python3 src/mobile/ci/release_assets.py checksums --dist dist --prefix "$ARTIFACT_PREFIX"
    python3 src/mobile/ci/release_assets.py check --dist dist --prefix "$ARTIFACT_PREFIX" \\
        --release-json "$RUNNER_TEMP/release.json"

* ``checksums`` writes ``<prefix>_mobile_SHA256SUMS.txt`` (``sha256sum`` format) over every
  other file in ``dist``.
* ``check`` refuses to publish unless ``dist`` holds exactly the mobile set (both Android ABIs,
  the unsigned IPA, the AltStore source and the checksums; the AAB and the signed IPA when
  present) under the names the app's update check knows
  (``glossarion_mobile.services.updates.classify_asset``), and no file would replace a release
  asset that is not a mobile asset. The desktop tag workflows publish to the same release, so
  the mobile job only ever creates the release or appends to it (it may replace its own mobile
  assets on a re-run, never a desktop file). ``--release-json`` is the GitHub API release object
  (``gh api repos/<repo>/releases/tags/<tag>``) or ``{}`` when the release does not exist yet.

Stdlib only; Python 3.10+. Exit status 1 on any problem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

CI_DIR = Path(__file__).resolve().parent
MOBILE_DIR = CI_DIR.parent
APP_DIR = MOBILE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from glossarion_mobile.services.updates import ALTSTORE_SOURCE_ASSET, classify_asset  # noqa: E402  (stdlib-only module)

__all__ = ["ANDROID_ABIS", "checksums_name", "clobber_problems", "expected_names", "main", "set_problems",
           "write_checksums"]

ANDROID_ABIS = ("arm64-v8a", "x86_64")


def checksums_name(prefix: str) -> str:
    return f"{prefix}_mobile_SHA256SUMS.txt"


def expected_names(prefix: str, *, debug_signed: bool, aab: bool, signed_ipa: bool) -> set:
    """The files one mobile release publishes (the names build-mobile.yml writes)."""
    suffix = "_debugsigned" if debug_signed else ""
    names = {f"{prefix}_Android_{abi}{suffix}.apk" for abi in ANDROID_ABIS}
    names |= {f"{prefix}_iOS_unsigned.ipa", ALTSTORE_SOURCE_ASSET, checksums_name(prefix)}
    if aab:
        names.add(f"{prefix}_Android.aab")
    if signed_ipa:
        names.add(f"{prefix}_iOS.ipa")
    return names


def set_problems(names: Iterable[str], prefix: str) -> list:
    """Problems with the files about to be published (``[]`` = a complete, mobile-only set)."""
    names = sorted(set(names))
    problems = [f"{name}: not a mobile release asset name" for name in names if classify_asset(name) is None]
    problems += [f"{name}: does not start with {prefix}_" for name in names
                 if name != ALTSTORE_SOURCE_ASSET and not name.startswith(prefix + "_")]
    debug = any(name.endswith("_debugsigned.apk") for name in names)
    expected = expected_names(prefix, debug_signed=debug, aab=f"{prefix}_Android.aab" in names,
                              signed_ipa=f"{prefix}_iOS.ipa" in names)
    problems += [f"{name}: missing" for name in sorted(expected - set(names))]
    problems += [f"{name}: unexpected" for name in names if name not in expected and classify_asset(name) is not None]
    signed = [n for n in names if n.endswith(".apk") and not n.endswith("_debugsigned.apk")]
    if debug and signed:
        problems.append("debug-signed and release-signed APKs mixed in one release")
    return problems


def clobber_problems(names: Iterable[str], release: Optional[dict]) -> list:
    """Files that would replace a release asset the mobile job does not own."""
    existing = {str(asset.get("name") or "") for asset in (release or {}).get("assets") or [] if isinstance(asset, dict)}
    return [f"{name}: would replace the release asset {name!r}, which is not a mobile asset"
            for name in sorted(set(names) & existing) if classify_asset(name) is None]


def write_checksums(dist: Path, prefix: str) -> Path:
    out = dist / checksums_name(prefix)
    lines = []
    for path in sorted((p for p in dist.iterdir() if p.is_file() and p.name != out.name), key=lambda p: p.name):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        lines.append(f"{digest.hexdigest()}  {path.name}\n")
    out.write_text("".join(lines), encoding="utf-8", newline="\n")
    return out


def _dist_names(dist: Path) -> list:
    return sorted(p.name for p in dist.iterdir() if p.is_file())


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sums = sub.add_parser("checksums", help="write <prefix>_mobile_SHA256SUMS.txt")
    sums.add_argument("--dist", type=Path, required=True)
    sums.add_argument("--prefix", required=True)
    check = sub.add_parser("check", help="complete mobile-only set; never replace a desktop asset")
    check.add_argument("--dist", type=Path, required=True)
    check.add_argument("--prefix", required=True)
    check.add_argument("--release-json", type=Path, default=None,
                       help="GitHub release object (gh api .../releases/tags/<tag>); missing or {} = no release yet")
    args = parser.parse_args(argv)

    if not args.dist.is_dir():
        print(f"release_assets: {args.dist} is not a directory", file=sys.stderr)
        return 1
    if args.command == "checksums":
        out = write_checksums(args.dist, args.prefix)
        sys.stdout.write(out.read_text(encoding="utf-8"))
        return 0

    names = _dist_names(args.dist)
    release: dict = {}
    if args.release_json is not None and args.release_json.is_file():
        text = args.release_json.read_text(encoding="utf-8").strip()
        release = json.loads(text) if text else {}
    problems = set_problems(names, args.prefix) + clobber_problems(names, release)
    for name in names:
        print(f"  {name}  ({classify_asset(name)})")
    if problems:
        for problem in problems:
            print(f"::error title=Mobile release assets::{problem}")
        return 1
    existing = len((release or {}).get("assets") or [])
    print(f"OK: {len(names)} mobile assets; the release {'has ' + str(existing) + ' assets' if release else 'does not exist yet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
