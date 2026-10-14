"""Manually publish verified job artifacts to an existing GitLab release.

Requires a protected tag, a matching artifact manifest, and the explicit
pipeline opt-in. Does not create releases, mint credentials, or contact GitHub.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import build_opener, HTTPRedirectHandler, Request

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {".exe", ".dmg", ".zip", ".apk", ".aab", ".ipa"}


class ReleaseError(RuntimeError):
    pass


def require(condition, message):
    if not condition: raise ReleaseError(message)


def prerequisites(env):
    require(env.get("GLOSSARION_PUBLISH") == "true", "Release publication was not explicitly requested.")
    require(env.get("CI_COMMIT_REF_PROTECTED") == "true", "Release requires a protected ref.")
    require(re.fullmatch(r"v[0-9][A-Za-z0-9_.-]*", env.get("CI_COMMIT_TAG", "")), "Release requires a version tag.")
    for name in ("CI_COMMIT_SHA", "CI_JOB_TOKEN", "CI_API_V4_URL", "CI_SERVER_URL", "CI_PROJECT_ID"):
        require(env.get(name), "Release prerequisite missing: " + name)
    api, server = urlparse(env["CI_API_V4_URL"]), urlparse(env["CI_SERVER_URL"])
    require(api.scheme == server.scheme == "https" and api.netloc == server.netloc and not api.username and not api.password,
            "Release API must use HTTPS on this GitLab server.")
    require(env["CI_PROJECT_ID"].isdigit(), "CI_PROJECT_ID must be numeric.")


def artifact_entries(root, commit):
    entries = []
    for manifest in sorted((root / "artifacts").glob("*/manifest.json")):
        data = json.loads(manifest.read_text())
        require(data.get("source_commit") == commit, "Artifact source commit does not match the release tag.")
        label = data.get("label", "")
        require(re.fullmatch(r"[a-z0-9-]+", label), "Invalid artifact label.")
        require(manifest.parent.name == label and not manifest.parent.is_symlink(), "Artifact directory does not match its label.")
        for item in data.get("artifacts", []):
            name = item.get("file", "")
            require(name and Path(name).name == name and "/" not in name and "\\" not in name, "Invalid artifact path.")
            path = manifest.parent / name
            require(path.suffix.lower() in ALLOWED and path.is_file() and not path.is_symlink(), "Expected release artifact is missing or unsupported.")
            require(path.resolve().parent == manifest.parent.resolve(), "Artifact escapes its manifest directory.")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            require(digest == item.get("sha256"), "Artifact digest changed after the build.")
            # Emulator/simulator ZIPs and debug APKs are diagnostic artifacts, not releases.
            require("simulator" not in name.lower() and "debugsigned" not in name.lower(),
                    "Simulator/debug-signed artifacts cannot be published by this release job.")
            entries.append((label, path, digest))
    require(entries, "No verified build artifacts; no release upload performed.")
    require(len({(label, path.name) for label, path, _ in entries}) == len(entries), "Duplicate release artifacts.")
    return entries


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ReleaseError("Release API redirected; refusing to forward the job token.")


def publish(env, root=ROOT, opener=None):
    prerequisites(env)
    entries = artifact_entries(root, env["CI_COMMIT_SHA"])
    opener = opener or build_opener(NoRedirect())
    api = env["CI_API_V4_URL"].rstrip("/") + "/projects/" + env["CI_PROJECT_ID"]
    tag = quote(env["CI_COMMIT_TAG"], safe="")
    headers = {"JOB-TOKEN": env["CI_JOB_TOKEN"]}

    def request(method, url, body=None, extra=None):
        req = Request(url, data=body, headers={**headers, **(extra or {})}, method=method)
        try:
            return opener.open(req, timeout=120)
        except HTTPError as exc:
            raise ReleaseError(f"Release API {method} failed (HTTP {exc.code}); no error body or credentials logged.") from None

    with request("GET", api + "/releases/" + tag) as response:
        release = json.load(response)
    require(release.get("tag_name") == env["CI_COMMIT_TAG"], "The existing release tag did not match.")
    require(release.get("commit", {}).get("id") == env["CI_COMMIT_SHA"], "The existing release points at a different commit.")
    existing_links = release.get("assets", {}).get("links", [])
    require(not existing_links, "Release already has asset links; review it manually instead of overwriting or duplicating assets.")
    # Preflight every package path before uploading any file. No replacement/clobber behavior.
    urls = []
    for label, path, _ in entries:
        package_name = label + "--" + path.name
        url = api + "/packages/generic/glossarion/" + tag + "/" + quote(package_name, safe="")
        try:
            with opener.open(Request(url, headers=headers, method="HEAD"), timeout=60):
                raise ReleaseError("A release package path already exists; refusing to replace it.")
        except HTTPError as exc:
            require(exc.code == 404, f"Package preflight failed (HTTP {exc.code}).")
        urls.append(url)
    for (label, path, _), url in zip(entries, urls):
        with path.open("rb") as stream:
            with request("PUT", url, stream, {"Content-Type": "application/octet-stream", "Content-Length": str(path.stat().st_size)}):
                pass
        payload = json.dumps({"name": path.name, "url": url, "link_type": "package"}).encode()
        with request("POST", api + "/releases/" + tag + "/assets/links", payload, {"Content-Type": "application/json"}):
            pass
        print("Published verified artifact: " + label + "/" + path.name)


if __name__ == "__main__":
    try:
        publish(dict(os.environ))
    except (ReleaseError, OSError, ValueError) as exc:
        print("FAILED: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
