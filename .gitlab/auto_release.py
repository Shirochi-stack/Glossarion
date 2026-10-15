"""Append verified tag-build outputs to an existing GitLab release.

Uses only the running job's token. Never creates tags/releases, edits release
notes, removes assets, contacts GitHub, or starts another pipeline. Re-running
this publisher reuses identical package files and links without rebuilding.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import sys
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import build_opener, HTTPRedirectHandler, Request

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"desktop-linux-standard": ".zip", "desktop-linux-nocuda": ".zip",
          "desktop-linux-turbolite": ".zip", "mobile-apk": ".apk", "mobile-ipa": ".ipa"}


class ReleaseError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise ReleaseError(message)


def digest(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def prerequisites(env, root):
    require(env.get("GLOSSARION_TASK") == "release-builds" and env.get("GLOSSARION_PUBLISH") == "true",
            "Automatic release publication was not selected.")
    require(env.get("CI_PIPELINE_SOURCE") == "pipeline"
            and env.get("GLOSSARION_RELEASE_EVENT_VALIDATED") == "true",
            "Automatic publication requires a validated release-event dispatch.")
    tag = env.get("CI_COMMIT_TAG", "")
    require(re.fullmatch(r"v[0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9_.+-]*)", tag), "A version tag is required.")
    require(env.get("CI_PROJECT_ID") == "87623444" and env.get("CI_SERVER_URL") == "https://gitlab.com"
            and env.get("CI_API_V4_URL") == "https://gitlab.com/api/v4", "Unexpected release project/server.")
    require(env.get("CI_JOB_TOKEN") and re.fullmatch(r"[a-f0-9]{40}", env.get("CI_COMMIT_SHA", "")),
            "The release job token/source commit is missing.")
    version = runpy.run_path(str(root / "src/app_version.py"))["VERSION_TAG"]
    require(version == tag, "The tag does not match the packaged source version.")


def artifact_entries(root, commit):
    entries = []
    for label, suffix in LABELS.items():
        folder = root / "artifacts" / label
        manifest = folder / "manifest.json"
        if not manifest.exists():
            continue
        require(not folder.is_symlink() and not manifest.is_symlink()
                and folder.resolve().is_relative_to(root.resolve()), "Unexpected artifact symlink.")
        data = json.loads(manifest.read_text())
        require(data.get("label") == label and data.get("source_commit") == commit,
                "Artifact source/label does not match this tag pipeline.")
        for item in data.get("artifacts", []):
            name = item.get("file", "")
            require(name and Path(name).name == name and not re.search(r'[<>:"/\\|?*%!\x00-\x1f]', name)
                    and not name.endswith((" ", ".")), "Invalid release artifact name.")
            path = folder / name
            require(path.suffix == suffix and "simulator" not in name.lower() and path.is_file()
                    and not path.is_symlink() and path.resolve().parent == folder.resolve(),
                    "Unexpected/missing release artifact.")
            require(path.stat().st_size > 0, "Empty release artifact.")
            with path.open("rb") as stream:
                value = digest(stream)
            require(value == item.get("sha256"), "Artifact SHA-256 differs from its verified build manifest.")
            entries.append((label, path, value))
    require(entries, "No verified Linux/APK/IPA output exists; no publication performed.")
    require(len({p.name for _, p, _ in entries}) == len(entries), "Duplicate release artifact names.")
    return entries


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ReleaseError("Release API redirected; refusing to forward the job token.")


def publish(env, root=ROOT, opener=None):
    root = Path(root)
    prerequisites(env, root)
    entries = artifact_entries(root, env["CI_COMMIT_SHA"])
    opener = opener or build_opener(NoRedirect())
    api = env["CI_API_V4_URL"] + "/projects/" + env["CI_PROJECT_ID"]
    tag = env["CI_COMMIT_TAG"]
    release_url = api + "/releases/" + quote(tag, safe="")
    headers = {"JOB-TOKEN": env["CI_JOB_TOKEN"]}

    def request(method, url, body=None, extra=None):
        return opener.open(Request(url, data=body, headers={**headers, **(extra or {})}, method=method), timeout=120)

    def release():
        with request("GET", release_url) as response:
            data = json.load(response)
        require(data.get("tag_name") == tag and data.get("commit", {}).get("id") == env["CI_COMMIT_SHA"],
                "The existing release points at a different tag/source commit.")
        return data

    current = release()
    uploads = []
    for label, path, value in entries:
        metadata = (json.dumps({"schema_version": 1, "tag": tag, "name": path.name,
                               "size": path.stat().st_size, "sha256": value,
                               "source_commit": env["CI_COMMIT_SHA"]}, indent=2) + "\n").encode()
        for name, content, checksum, size in ((path.name, path, value, path.stat().st_size),
                (path.name + ".sha256.json", metadata, hashlib.sha256(metadata).hexdigest(), len(metadata))):
            url = api + "/packages/generic/glossarion/" + quote(tag, safe="") + "/" + quote(label + "--" + name, safe="")
            uploads.append({"name": name, "url": url, "content": content, "sha256": checksum, "size": size})

    def check_link(item, links):
        matches = [link for link in links if link.get("name") == item["name"] or link.get("url") == item["url"]]
        require(all(link.get("name") == item["name"] and link.get("url") == item["url"] for link in matches),
                "A release asset name/URL already belongs to different content; preserving it.")
        return bool(matches)

    def package_exists(item):
        try:
            with request("HEAD", item["url"]) as response:
                checksum = response.headers.get("X-Checksum-SHA256")
                size = response.headers.get("Content-Length")
            if checksum and size:
                require(checksum == item["sha256"] and size == str(item["size"]),
                        "Existing package differs; refusing replacement or duplicate upload.")
            else:
                with request("GET", item["url"]) as response:
                    require(digest(response) == item["sha256"], "Existing package differs; refusing replacement.")
            return True
        except HTTPError as exc:
            if exc.code == 404:
                return False
            raise

    # Complete conflict preflight before any write. Windows/other assets and
    # the existing release title/notes are never changed by this publisher.
    for item in uploads:
        check_link(item, current.get("assets", {}).get("links", []))
        item["exists"] = package_exists(item)
    for item in uploads:
        if not item["exists"]:
            content = item["content"]
            with (content.open("rb") if isinstance(content, Path) else io.BytesIO(content)) as stream:
                with request("PUT", item["url"], stream, {"Content-Type": "application/octet-stream",
                             "Content-Length": str(item["size"])}):
                    pass
            require(package_exists(item), "Uploaded package is not readable/verifiable yet; retry only this publisher.")
    # Sidecar first: consumers never see a new binary link without checksum metadata.
    for item in sorted(uploads, key=lambda entry: not entry["name"].endswith(".sha256.json")):
        if check_link(item, current.get("assets", {}).get("links", [])):
            continue
        payload = json.dumps({"name": item["name"], "url": item["url"], "link_type": "package"}).encode()
        try:
            with request("POST", release_url + "/assets/links", payload, {"Content-Type": "application/json"}):
                pass
        except HTTPError as exc:
            if exc.code not in (400, 409):
                raise
            # Another coordinated publisher may have added the same link.
            current = release()
            require(check_link(item, current.get("assets", {}).get("links", [])), "Release link creation failed.")
        current.setdefault("assets", {}).setdefault("links", []).append({"name": item["name"], "url": item["url"]})
        print("Attached verified release asset: " + item["name"])
    return {"tag": tag, "source_commit": env["CI_COMMIT_SHA"], "artifacts": len(entries)}


if __name__ == "__main__":
    try:
        print(json.dumps(publish(dict(os.environ))))
    except HTTPError as exc:
        print(f"FAILED: Release API returned HTTP {exc.code}; retry this publisher after resolving it. No credentials/body logged.", file=sys.stderr)
        raise SystemExit(1)
    except (ReleaseError, OSError, ValueError) as exc:
        print("FAILED: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
