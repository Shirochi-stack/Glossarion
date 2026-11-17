#!/usr/bin/env python3
"""Index wheels held at an older build (index_holds.toml): put them into the wheelhouse.

    add          after `wheels.py assemble`: download each held wheel of the platform from
                 pypi.flet.dev (found by its file name on the index page, sha256 from
                 index_holds.toml), change only its build tag to the manifest's build_tag
                 (wheels.py retag_wheel, proven with compare_retag), write it into the wheelhouse
                 and list it in SHA256SUMS
    provenance   after `wheels.py provenance`: record where the held wheels come from in the
                 wheelhouse's provenance.json

Kept apart from wheels.py and wheelhouse.toml, which are wheel cache keys in
.github/workflows/mobile-wheels.yml: a hold comes and goes with the state of the index and must
not rebuild the wheels we compile. Stdlib only.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wheels  # noqa: E402  (download_verified, retag_wheel, compare_retag; puts tools/ on sys.path)

cmw = wheels.cmw
# In --downloads: what `add` fetched, for `provenance`.
SOURCES_NAME = "held-sources.json"


def index_url(fetcher, w, target: str) -> str:
    """The pypi.flet.dev URL of the held build of ``w`` for ``target`` (exactly one file of that name)."""
    name = w.official_filename(target)
    page = f"{cmw.FLET_INDEX}/{w.name}/"
    urls = sorted({urllib.parse.urljoin(page, f.url) for f in fetcher.files(w.name)
                   if f.source == "flet" and f.filename == name and f.url})
    if len(urls) != 1:
        raise wheels.WheelsError(f"{page}: expected one {name}, found {len(urls)}")
    parts = urllib.parse.urlsplit(urls[0])
    if parts.scheme != "https" or parts.hostname != urllib.parse.urlsplit(cmw.FLET_INDEX).hostname:
        raise wheels.WheelsError(f"{page}: {name} links to {urls[0]}, which is not on {cmw.FLET_INDEX}")
    return urls[0]


def update_sums(wheelhouse: Path, added: dict) -> None:
    """SHA256SUMS with the lines of ``added`` ({file name: sha256}); the other lines are kept."""
    sums = wheelhouse / "SHA256SUMS"
    listed = {}
    if sums.is_file():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, _, name = line.strip().partition(" ")
                listed[name.lstrip(" *")] = digest
    listed.update(added)
    sums.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(listed.items())),
                    encoding="utf-8", newline="\n")


def cmd_add(manifest, args) -> None:
    items = manifest.held_targets(args.platform)
    if not items:
        wheels.log(f"no held index wheels for {args.platform}")
        return
    wheelhouse, downloads, tag = Path(args.wheelhouse), Path(args.downloads), manifest.build_tag
    if not wheelhouse.is_dir():
        raise wheels.WheelsError(f"{wheelhouse}: not a directory (run wheels.py assemble first)")
    fetcher = cmw.Fetcher()
    sources, rows = {}, []
    for w, target in items:
        sha = w.retag_source(target)[1]
        url = index_url(fetcher, w, target)
        original = downloads / w.official_filename(target)
        wheels.download_verified(url, original, sha)
        dest = wheelhouse / w.filename(target, tag)
        wheels.retag_wheel(original, dest, tag)
        errors = wheels.compare_retag(original, dest, tag)
        if errors:
            dest.unlink(missing_ok=True)
            raise wheels.WheelsError("; ".join(errors))
        digest = cmw.file_sha256(dest)
        sources[dest.name] = {"package": w.name, "version": w.version, "target": target, "url": url,
                              "file": original.name, "sha256": sha, "index_build": int(w.held_build),
                              "build_tag": tag, "retagged_sha256": digest}
        rows.append(f"| `{dest.name}` | {dest.stat().st_size:,} | `{digest}` | pypi.flet.dev {original.name} "
                    f"(sha256 {sha}), held |")
        wheels.log(f"held {original.name} (sha256 {sha}) -> {dest.name} (sha256 {digest}, build {tag})")
    update_sums(wheelhouse, {name: s["retagged_sha256"] for name, s in sources.items()})
    downloads.mkdir(parents=True, exist_ok=True)
    (downloads / SOURCES_NAME).write_text(json.dumps(sources, indent=1) + "\n", encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"## Held index wheels ({args.platform})\n\n| wheel | bytes | sha256 | from |\n|---|---|---|---|\n")
            f.write("\n".join(rows) + "\n\n")


def cmd_provenance(manifest, args) -> None:
    items = manifest.held_targets(args.platform)
    if not items:
        wheels.log(f"no held index wheels for {args.platform}")
        return
    wheelhouse = Path(args.wheelhouse)
    path = wheelhouse / "provenance.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    sources = json.loads((Path(args.downloads) / SOURCES_NAME).read_text(encoding="utf-8"))
    entries = {e.get("file"): e for e in doc.get("wheels", [])}
    for w, target in items:
        name = w.filename(target, manifest.build_tag)
        entry, source = entries.get(name), sources.get(name)
        if entry is None or source is None:
            raise wheels.WheelsError(f"{name}: not in {path.name} or not added in this run "
                                     f"(run index_holds.py add, then wheels.py provenance)")
        if entry.get("sha256") != source["retagged_sha256"]:
            raise wheels.WheelsError(f"{name}: {path.name} has sha256 {entry.get('sha256')}, add wrote "
                                     f"{source['retagged_sha256']}")
        entry["package"] = w.name
        entry["origin"] = (f"pypi.flet.dev {source['file']} (sha256 {source['sha256']}), held at index build "
                           f"{w.held_build}; only the build tag changed")
        entry["built_in_this_run"] = False
        entry["retagged_in_this_run"] = True
        entry["held"] = {k: source[k] for k in ("url", "file", "sha256", "index_build", "build_tag")}
    doc["index_holds"] = {
        "path": "src/mobile/ci/wheels/" + cmw.INDEX_HOLDS_NAME,
        "sha256": cmw.file_sha256(manifest.path.parent / cmw.INDEX_HOLDS_NAME),
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    wheels.log(f"recorded {len(items)} held index wheels in {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Index wheels held at an older build (see docstring).")
    ap.add_argument("--manifest", default=str(wheels.DEFAULT_MANIFEST),
                    help=f"wheelhouse.toml ({cmw.INDEX_HOLDS_NAME} is read from its directory)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_text in (("add", "fetch, retag and add the held wheels to the wheelhouse"),
                            ("provenance", "record the held wheels in the wheelhouse's provenance.json")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--platform", required=True, choices=wheels.PLATFORMS)
        p.add_argument("--wheelhouse", required=True, help="the assembled wheelhouse directory")
        p.add_argument("--downloads", required=True, help="where the index's original wheels are kept")
    args = ap.parse_args(argv)
    try:
        manifest = cmw.load_wheelhouse_manifest(args.manifest)
        (cmd_add if args.cmd == "add" else cmd_provenance)(manifest, args)
    except (wheels.WheelsError, cmw.ManifestError, OSError, RuntimeError, ValueError, KeyError) as e:
        print(f"index_holds: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
