#!/usr/bin/env python3
"""Where an APK's bytes go: native libraries, the embedded-Python zips and the rest.

Flet 1.0.3 / serious_python package an Android app as
  * ``lib/<abi>/*.so``: the Flutter engine, CPython and every extension module (relocated as
    ``lib<mangled>.so``). With the default (modern) packaging they are stored uncompressed and
    page-aligned, so they cost their full size in the APK; ``--android-legacy-packaging``
    compresses them (smaller APK download) and Android extracts them at install (bigger install).
  * ``assets/**/{stdlib,sitepackages,extract}.zip`` and the app archive: STORED zips (zipimport
    reads them without zlib), so the .pyc and data inside cost their full size too.

This report lists the totals per category, the largest native libraries, the largest packages
inside the stored zips, and what the APK would weigh with compressed native libraries
(zlib level 6, the deflate Android's packager uses), so the packaging trade-off is measured,
not guessed. It never fails the build (exit status 1 only for an unreadable APK)::

    python3 src/mobile/ci/apk_size_report.py dist/*.apk --json build/apk_size.json \\
        [--summary "$GITHUB_STEP_SUMMARY"] [--top 25]

Stdlib only; Python 3.10+.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Optional

__all__ = ["analyze_apk", "main", "markdown"]

MB = 1024 * 1024


def _category(name: str) -> str:
    if name.startswith("lib/") and name.endswith(".so"):
        return "native libraries (lib/<abi>/*.so)"
    if name.endswith(".zip") and name.startswith("assets/"):
        return "embedded Python zips (assets/**.zip)"
    if name.startswith("assets/flutter_assets/"):
        return "Flutter assets"
    if name.endswith(".dex"):
        return "dex (Java/Kotlin)"
    if name.startswith("res/") or name == "resources.arsc":
        return "Android resources"
    return "other"


def _deflated_size(data: bytes) -> int:
    return len(zlib.compress(data, 6))


def _zip_packages(data: bytes) -> dict:
    """Bytes per top-level entry (package / module) inside a nested zip."""
    out: dict = {}
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as inner:
            for info in inner.infolist():
                if info.is_dir():
                    continue
                top = PurePosixPath(info.filename).parts[0]
                if top.endswith(".dist-info"):
                    top = "*.dist-info"
                out[top] = out.get(top, 0) + info.file_size
    except zipfile.BadZipFile:
        pass
    return out


def analyze_apk(path: Path, *, top: int = 25, estimate_legacy: bool = True) -> dict:
    report: dict = {"apk": path.name, "size": path.stat().st_size, "categories": {}, "native": [], "zips": {},
                    "packages": []}
    native_stored = native_deflated = 0
    packages: dict = {}
    with zipfile.ZipFile(path) as apk:
        for info in apk.infolist():
            if info.is_dir():
                continue
            cat = report["categories"].setdefault(_category(info.filename), {"files": 0, "stored": 0, "unpacked": 0})
            cat["files"] += 1
            cat["stored"] += info.compress_size
            cat["unpacked"] += info.file_size
            if info.filename.startswith("lib/") and info.filename.endswith(".so"):
                entry = {"name": info.filename, "size": info.file_size, "in_apk": info.compress_size,
                         "compressed": info.compress_type != zipfile.ZIP_STORED}
                if estimate_legacy:
                    data = apk.read(info)
                    entry["deflated"] = _deflated_size(data) if not entry["compressed"] else info.compress_size
                    native_deflated += entry["deflated"]
                native_stored += info.compress_size
                report["native"].append(entry)
            elif info.filename.startswith("assets/") and info.filename.endswith(".zip"):
                data = apk.read(info)
                report["zips"][info.filename] = {"size": info.file_size, "in_apk": info.compress_size}
                for name, size in _zip_packages(data).items():
                    packages[name] = packages.get(name, 0) + size
    report["native"].sort(key=lambda e: -e["size"])
    report["native_total_in_apk"] = native_stored
    if estimate_legacy:
        report["native_total_deflated"] = native_deflated
        report["legacy_packaging_estimate"] = report["size"] - native_stored + native_deflated
    report["packages"] = sorted(({"name": k, "size": v} for k, v in packages.items()), key=lambda e: -e["size"])[:top]
    report["native"] = report["native"][:top]
    return report


def _mb(value: int) -> str:
    return f"{value / MB:.1f} MB"


def markdown(reports: list) -> str:
    lines = ["## APK size", ""]
    for rep in reports:
        lines += [f"### {rep['apk']} — {_mb(rep['size'])}", "", "| Category | Files | In APK | Unpacked |", "|---|---:|---:|---:|"]
        for name, cat in sorted(rep["categories"].items(), key=lambda kv: -kv[1]["stored"]):
            lines.append(f"| {name} | {cat['files']} | {_mb(cat['stored'])} | {_mb(cat['unpacked'])} |")
        if "legacy_packaging_estimate" in rep:
            lines += ["", f"Native libraries: {_mb(rep['native_total_in_apk'])} in the APK, "
                          f"{_mb(rep['native_total_deflated'])} deflated. With `--android-legacy-packaging` the APK would be "
                          f"about **{_mb(rep['legacy_packaging_estimate'])}** (the installed app then also keeps the "
                          f"extracted libraries, about {_mb(rep['native_total_in_apk'])} more on the device)."]
        lines += ["", "<details><summary>Largest native libraries</summary>", "", "| Library | Size | In APK |", "|---|---:|---:|"]
        lines += [f"| `{e['name']}` | {_mb(e['size'])} | {_mb(e['in_apk'])} |" for e in rep["native"]]
        lines += ["", "</details>", "", "<details><summary>Largest packages in the stored Python zips</summary>", "",
                  "| Package | Size |", "|---|---:|"]
        lines += [f"| `{e['name']}` | {_mb(e['size'])} |" for e in rep["packages"]]
        lines += ["", "</details>", ""]
    return "\n".join(lines) + "\n"


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("apks", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None, help="append the Markdown report (GITHUB_STEP_SUMMARY)")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--no-legacy-estimate", action="store_true", help="skip deflating the native libraries")
    args = parser.parse_args(argv)
    reports = []
    rc = 0
    for apk in args.apks:
        if apk.suffix.lower() != ".apk":
            continue
        try:
            reports.append(analyze_apk(apk, top=args.top, estimate_legacy=not args.no_legacy_estimate))
        except (OSError, zipfile.BadZipFile) as exc:
            print(f"apk_size_report: {apk}: {exc}", file=sys.stderr)
            rc = 1
    text = markdown(reports)
    print(text)
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
