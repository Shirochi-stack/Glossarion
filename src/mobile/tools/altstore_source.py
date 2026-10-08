#!/usr/bin/env python3
"""Write the AltStore / SideStore source (``altstore-source.json``) for a mobile release.

The gated release job of ``.github/workflows/build-mobile.yml`` runs this (it used to be an
inline heredoc there). The source lists one app version that points at the **unsigned** IPA
release asset, which AltStore / SideStore re-sign on the device::

    python3 src/mobile/tools/altstore_source.py \\
        --ipa dist/Glossarion_v9.15.0_iOS_unsigned.ipa --tag v9.15.0 \\
        --repo Shirochi-stack/Glossarion --out dist/altstore-source.json

Everything comes from the IPA's ``Payload/<App>.app/Info.plist`` (bundle id, version, build
number, minimum iOS, the ``NS*UsageDescription`` privacy strings) and the arguments:

* ``downloadURL`` = ``https://github.com/<repo>/releases/download/<tag>/<ipa name>``;
* ``sourceURL`` = ``https://github.com/<repo>/releases/latest/download/altstore-source.json``
  (``--source-url`` overrides it: the "latest" release is often desktop-only, in which case
  AltStore's refresh of that URL fails until a mobile release is the latest again);
* ``iconURL`` = the committed ``src/mobile/app/assets/icon.png`` at the tag;
* ``date`` = now in UTC (``--date`` pins it, e.g. for tests).

Stdlib only; Python 3.10+. Exit status 1 when the IPA has no app Info.plist.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import plistlib
import sys
import zipfile
from pathlib import Path
from typing import Any, Optional

__all__ = ["SOURCE_FILE_NAME", "build_source", "default_source_url", "main", "read_info_plist", "write_source"]

SOURCE_FILE_NAME = "altstore-source.json"
APP_NAME = "Glossarion"
DEVELOPER = "Glossarion contributors"
SUBTITLE = "AI novel and manga translator"
DESCRIPTION = "Translate novels, EPUBs, PDFs and manga with your own AI models."
TINT = "E18F98"
CATEGORY = "utilities"
ICON_PATH = "src/mobile/app/assets/icon.png"


class SourceError(Exception):
    """The IPA cannot describe an AltStore app (no app Info.plist, missing keys)."""


def read_info_plist(ipa: Path) -> dict:
    """The app's ``Payload/<App>.app/Info.plist`` (XML or binary) from an IPA."""
    try:
        with zipfile.ZipFile(ipa) as archive:
            names = [
                name for name in archive.namelist()
                if name.startswith("Payload/") and name.endswith(".app/Info.plist") and name.count("/") == 2
            ]
            if not names:
                raise SourceError(f"{ipa}: no Payload/<App>.app/Info.plist")
            return plistlib.loads(archive.read(names[0]))
    except zipfile.BadZipFile as exc:
        raise SourceError(f"{ipa}: not a zip archive ({exc})") from exc


def default_source_url(repo: str) -> str:
    return f"https://github.com/{repo}/releases/latest/download/{SOURCE_FILE_NAME}"


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_source(info: dict, *, ipa_name: str, ipa_size: int, tag: str, repo: str, date: Optional[str] = None,
                 source_url: Optional[str] = None) -> dict:
    """The AltStore source document (same layout the inline release step wrote)."""
    for key in ("CFBundleIdentifier", "CFBundleShortVersionString", "CFBundleVersion"):
        if not info.get(key):
            raise SourceError(f"Info.plist has no {key}")
    releases = f"https://github.com/{repo}/releases"
    privacy = {key: value for key, value in info.items()
               if key.startswith("NS") and key.endswith("UsageDescription")}
    return {
        "name": APP_NAME,
        "identifier": f"{info['CFBundleIdentifier']}.source",
        "sourceURL": source_url or default_source_url(repo),
        "website": f"https://github.com/{repo}",
        "apps": [{
            "name": APP_NAME,
            "bundleIdentifier": info["CFBundleIdentifier"],
            "developerName": DEVELOPER,
            "subtitle": SUBTITLE,
            "localizedDescription": DESCRIPTION,
            "iconURL": f"https://raw.githubusercontent.com/{repo}/{tag}/{ICON_PATH}",
            "tintColor": TINT,
            "category": CATEGORY,
            "appPermissions": {"entitlements": [], "privacy": privacy},
            "versions": [{
                "version": info["CFBundleShortVersionString"],
                "buildVersion": str(info["CFBundleVersion"]),
                "date": date or _utc_now(),
                "localizedDescription": f"{APP_NAME} {tag}",
                "downloadURL": f"{releases}/download/{tag}/{ipa_name}",
                "size": int(ipa_size),
                "minOSVersion": str(info.get("MinimumOSVersion", "13.0")),
            }],
        }],
        "news": [],
    }


def write_source(ipa: Path, out: Path, *, tag: str, repo: str, date: Optional[str] = None,
                 source_url: Optional[str] = None) -> dict:
    source = build_source(read_info_plist(ipa), ipa_name=ipa.name, ipa_size=ipa.stat().st_size, tag=tag, repo=repo,
                          date=date, source_url=source_url)
    text = json.dumps(source, indent=2) + "\n"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))  # LF on every platform
    os.replace(tmp, out)
    return source


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ipa", required=True, type=Path, help="the unsigned IPA release asset")
    parser.add_argument("--tag", required=True, help="release tag, e.g. v9.15.0")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""), help="owner/repo")
    parser.add_argument("--out", type=Path, default=None, help=f"output file (default: <ipa dir>/{SOURCE_FILE_NAME})")
    parser.add_argument("--date", default=None, help="version date (default: now, UTC)")
    parser.add_argument("--source-url", default=None, help="sourceURL (default: the latest release's asset)")
    args = parser.parse_args(argv)
    if not args.repo or "/" not in args.repo:
        parser.error("--repo owner/repo is required (or GITHUB_REPOSITORY)")
    if not args.tag:
        parser.error("--tag is required")
    out = args.out or args.ipa.with_name(SOURCE_FILE_NAME)
    try:
        source: Any = write_source(args.ipa, out, tag=args.tag, repo=args.repo, date=args.date,
                                   source_url=args.source_url)
    except (OSError, SourceError, plistlib.InvalidFileException) as exc:
        print(f"altstore_source: {exc}", file=sys.stderr)
        return 1
    version = source["apps"][0]["versions"][0]
    print(f"wrote {out} ({source['apps'][0]['bundleIdentifier']} {version['version']} ({version['buildVersion']}) "
          f"-> {version['downloadURL']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
