#!/usr/bin/env python3
"""Dependency-graph snapshot of the mobile app, built from ``src/mobile/uv.lock``.

Stdlib only (``tomllib``, Python 3.11+). The mobile-dependency-graph workflow posts the
JSON to the dependency submission API on every push to main that changes the lock, so
GitHub's dependency graph and Dependabot alerts follow the versions that ship.

Why a submission at all: an older user-submitted snapshot (detector
``glossarion-mobile-uv-lock``, correlator unknown) still claims ``src/mobile/uv.lock`` and
``src/mobile/pyproject.toml`` with versions from before several pins were bumped
(pypdf 6.19.0, certifi 2026.2.25), and user submissions outrank Dependabot's own parse
of the lock. Snapshots cannot be deleted. Among user submissions from different
detectors GitHub uses the one whose correlator sorts first, so ``CORRELATOR`` starts with
``0`` and this snapshot takes over both manifests.

Every locked registry package is listed for both manifests with its exact version;
packages the project itself depends on (the virtual ``glossarion`` root's
``dependencies``, ``optional-dependencies`` and ``dev-dependencies``) are ``direct``,
the rest ``indirect``. The root and the in-repo editable extension are left out.

Usage::

    python tools/dependency_snapshot.py --sha <commit> --ref refs/heads/main [--job-id <run id>]
    python tools/dependency_snapshot.py --sha <commit> --ref <ref> --out snapshot.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import tomllib
from pathlib import Path

MOBILE_DIR = Path(__file__).resolve().parent.parent
LOCK_FILE = MOBILE_DIR / "uv.lock"
REPO_LOCK_PATH = "src/mobile/uv.lock"
REPO_PYPROJECT_PATH = "src/mobile/pyproject.toml"

CORRELATOR = "0-glossarion-mobile-uv-lock"
DETECTOR = {
    "name": "glossarion-mobile-dependency-snapshot",
    "version": "1",
    "url": "https://github.com/Shirochi-stack/Glossarion/blob/main/src/mobile/tools/dependency_snapshot.py",
}


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _root_package(packages: list[dict]) -> dict:
    for pkg in packages:
        if "virtual" in pkg.get("source", {}):
            return pkg
    raise SystemExit(f"{LOCK_FILE}: no virtual root package")


def _direct_names(root: dict) -> set[str]:
    names = {d["name"] for d in root.get("dependencies", [])}
    for group in ("optional-dependencies", "dev-dependencies"):
        for deps in root.get(group, {}).values():
            names.update(d["name"] for d in deps)
    return {_normalize(n) for n in names}


def resolved_packages(lock: dict) -> dict:
    packages = lock.get("package", [])
    direct = _direct_names(_root_package(packages))
    resolved: dict[str, dict] = {}
    for pkg in packages:
        if "registry" not in pkg.get("source", {}):
            continue  # virtual root, editable in-repo extension
        name = _normalize(pkg["name"])
        purl = f"pkg:pypi/{name}@{pkg['version']}"
        resolved[purl] = {
            "package_url": purl,
            "relationship": "direct" if name in direct else "indirect",
            "scope": "runtime",
        }
    return dict(sorted(resolved.items()))


def build_snapshot(lock: dict, sha: str, ref: str, job_id: str) -> dict:
    resolved = resolved_packages(lock)
    return {
        "version": 0,
        "sha": sha,
        "ref": ref,
        "job": {"correlator": CORRELATOR, "id": job_id},
        "detector": DETECTOR,
        "scanned": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifests": {
            path: {"name": path, "file": {"source_location": path}, "resolved": resolved}
            for path in (REPO_LOCK_PATH, REPO_PYPROJECT_PATH)
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--sha", required=True, help="commit the snapshot describes (40 hex)")
    parser.add_argument("--ref", required=True, help="e.g. refs/heads/main")
    parser.add_argument("--job-id", default="local", help="CI run id")
    parser.add_argument("--out", type=Path, help="write here instead of stdout")
    args = parser.parse_args(argv)

    with LOCK_FILE.open("rb") as fh:
        snapshot = build_snapshot(tomllib.load(fh), args.sha, args.ref, args.job_id)
    text = json.dumps(snapshot, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
